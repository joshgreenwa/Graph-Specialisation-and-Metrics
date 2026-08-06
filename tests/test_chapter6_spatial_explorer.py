import json
from pathlib import Path

import matplotlib
import numpy as np
import pytest
import torch

from graph_specialisation_metrics.chapter6_spatial_explorer import (
    head_metrics,
    layer_summary,
    load_models,
    run,
)

matplotlib.use("Agg")


def _score() -> dict:
    axis = (0, 1, 2)
    semantic = np.asarray(
        [
            [[1.0, 0.0, 0.0], [0.5, 0.0, 0.5]],
            [[0.0, 1.0, 0.0], [0.0, 0.5, 0.5]],
        ]
    )
    structural = np.asarray(
        [
            [[0.0, 1.0, 0.0], [0.5, 0.0, 0.5]],
            [[0.0, 0.0, 1.0], [0.0, 0.5, 0.5]],
        ]
    )

    def channel(values: np.ndarray, offset: float) -> dict:
        graph_zero = values.copy()
        graph_one = values.copy()
        graph_one[..., 0] += offset
        raw = values.sum(axis=-1)
        point = np.zeros((2, values.shape[0] + 1, len(axis)))
        point[0, :-1] = values.sum(axis=1)
        point[0, -1] = values.sum(axis=(0, 1))
        point[1] = point[0]
        return {
            "raw": raw,
            "heatmap_exact_head": values,
            "heatmap_per_opportunity_head": values,
            "graph_distance_contribution": {10: graph_zero, 11: graph_one},
            "distance_intervals": {
                "estimate": point,
                "low": point * 0.9,
                "high": point * 1.1,
            },
            "distance_support": {"reportable": np.ones(len(axis), dtype=bool)},
        }

    raw_semantic = semantic.sum(axis=-1)
    raw_structural = structural.sum(axis=-1)
    joint = raw_semantic + raw_structural
    return {
        "axis": axis,
        "channels": {
            "semantic": channel(semantic, 0.2),
            "structural": channel(structural, 0.1),
        },
        "coordinates": {
            "joint_sensitivity": joint,
            "selectivity": (raw_semantic - raw_structural) / joint,
        },
        "clean_attention_distance": np.full_like(semantic, 1.0 / len(axis)),
    }


def _carriage() -> dict:
    rows = [
        {
            "seed": 42,
            "graph_id": 0,
            "source": 0,
            "donor": 1,
            "distance": distance,
            "carrier_kind": "molecular_node",
            "F_sens": value,
        }
        for distance, value in ((0.0, 1.0), (1.0, 2.0), (2.0, 1.0))
    ]
    return {"channels": {channel: {"pairs": rows} for channel in ("semantic", "structural")}}


def _write_artifact(root: Path, task: str, *, with_carriage: bool = True) -> None:
    task_dir = root / task / "seed_42"
    score_path = task_dir / "cache/scores/raw.pt"
    score_path.parent.mkdir(parents=True)
    torch.save(
        {
            "metadata": {
                "protocol_version": "donor-swap-specialisation-carriage-v4",
                "contract": {"task": task},
            },
            "value": _score(),
        },
        score_path,
    )
    (task_dir / "model.json").write_text(
        json.dumps({"train_seed": 42, "test_metric": 0.1}), encoding="utf-8"
    )
    if with_carriage:
        carriage_path = task_dir / "cache/carriage/fields.pt"
        carriage_path.parent.mkdir(parents=True)
        torch.save({"metadata": {}, "value": _carriage()}, carriage_path)


def test_head_metrics_keep_width_uncertainty_and_attention_separate(tmp_path):
    _write_artifact(tmp_path, "zinc_1hop")
    models, warnings = load_models([tmp_path], ["zinc_1hop"], seed=42)
    assert not warnings
    rows = head_metrics(models)
    first = next(row for row in rows if row["layer"] == 0 and row["head"] == 0)
    assert first["semantic_expected_distance"] == pytest.approx(0.0)
    assert first["structural_expected_distance"] == pytest.approx(1.0)
    assert first["semantic_spatial_variance"] == pytest.approx(0.0)
    assert first["attention_expected_distance"] == pytest.approx(1.0)
    assert first["overlap"] == pytest.approx(0.0)
    assert np.isfinite(first["semantic_expected_distance_sem"])

    summary = layer_summary(rows)
    sources = {row["source"] for row in summary}
    assert sources == {"semantic", "structural", "attention"}


def test_run_skips_missing_components_and_writes_exploratory_outputs(tmp_path):
    _write_artifact(tmp_path, "zinc_1hop")
    output = tmp_path / "output"
    result = run(
        [tmp_path],
        output,
        tasks=("zinc_1hop", "zinc_2hop"),
        seed=42,
        verbose=False,
    )
    assert result["tasks_loaded"] == ["zinc_1hop"]
    assert any("zinc_2hop" in warning for warning in result["warnings"])
    for name in (
        "cache_inventory.csv",
        "head_spatial_metrics.csv",
        "layer_spatial_summary.csv",
        "representative_heads.csv",
        "model_distance_profiles.csv",
        "score_profile_uncertainty.csv",
        "summary.json",
    ):
        assert (output / name).is_file()
    pngs = sorted((output / "figures").glob("*.png"))
    assert {path.name for path in pngs} == {
        "01_expected_distance_heatmaps.png",
        "02_peak_distance_heatmaps.png",
        "03_alignment_heatmaps.png",
        "04_layerwise_score_attention_distance.png",
        "05_spatial_width_and_uncertainty.png",
        "06_representative_head_profiles.png",
        "07_score_and_final_state_response.png",
    }
