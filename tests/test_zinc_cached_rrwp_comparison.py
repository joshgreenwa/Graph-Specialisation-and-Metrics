from pathlib import Path

import numpy as np

from graph_specialisation_metrics.methodology.bootstrap import Interval
from graph_specialisation_metrics.methodology.cache import ReadOnlyCacheArtifact
from graph_specialisation_metrics.zinc_cached_rrwp_comparison import (
    CachedModel,
    cache_inventory,
    carriage_profile,
    group_distance,
    run,
    score_interval_width_profile,
    score_profile,
)


def _score(axis=(0, 1, 2, 3, 4, 8, "virtual")):
    layers, heads, distances = 2, 2, len(axis)
    semantic_base = np.linspace(1.0, 0.2, distances)
    structural_base = np.linspace(0.2, 1.0, distances)
    factors = np.asarray([[1.0, 1.2], [0.8, 1.4]])[..., None]
    exact_semantic = factors * semantic_base
    exact_structural = factors * structural_base

    def channel(exact):
        raw = exact.sum(axis=-1)
        point = np.zeros((2, layers + 1, distances), dtype=np.float64)
        point[0, :-1] = exact.sum(axis=1)
        point[0, -1] = exact.sum(axis=(0, 1))
        point[1] = point[0]
        return {
            "raw": raw,
            "heatmap_exact_head": exact,
            "distance_intervals": Interval(
                estimate=point,
                low=point * 0.9,
                high=point * 1.1,
                replicates=100,
                rng_seed=7,
                resampled_levels=("graph",),
            ),
            "distance_support": {
                "reportable": np.ones(distances, dtype=bool),
            },
        }

    semantic = channel(exact_semantic)
    structural = channel(exact_structural)
    raw_semantic = semantic["raw"]
    raw_structural = structural["raw"]
    joint = raw_semantic + raw_structural
    return {
        "axis": axis,
        "coordinates": {
            "raw_semantic": raw_semantic,
            "raw_structural": raw_structural,
            "normalized_semantic": raw_semantic / joint,
            "normalized_structural": raw_structural / joint,
            "joint_sensitivity": joint,
            "selectivity": (raw_semantic - raw_structural) / joint,
            "active": np.ones((layers, heads), dtype=bool),
            "estimable": True,
        },
        "channels": {"semantic": semantic, "structural": structural},
    }


def _carriage():
    rows = [
        {
            "seed": 0,
            "graph_id": 10,
            "source": 2,
            "donor": 8,
            "distance": 0.0,
            "carrier_kind": "molecular_node",
            "F_sens": 1.0,
        },
        {
            "seed": 0,
            "graph_id": 10,
            "source": 2,
            "donor": 8,
            "distance": 1.0,
            "carrier_kind": "molecular_node",
            "F_sens": 3.0,
        },
        {
            "seed": 0,
            "graph_id": 10,
            "source": 3,
            "donor": 9,
            "distance": 0.0,
            "carrier_kind": "molecular_node",
            "F_sens": 2.0,
        },
        {
            "seed": 0,
            "graph_id": 10,
            "source": 3,
            "donor": 9,
            "distance": 1.0,
            "carrier_kind": "molecular_node",
            "F_sens": 2.0,
        },
    ]
    return {"channels": {channel: {"pairs": rows} for channel in ("semantic", "structural")}}


def _model(tmp_path: Path, task: str, *, include_virtual: bool) -> CachedModel:
    axis = (0, 1, 2, 3, 4, 8, "virtual") if include_virtual else (0, 1, 2, 3, 4, 8)
    score = _score(axis)
    score_artifact = ReadOnlyCacheArtifact(
        path=tmp_path / task / "raw.pt",
        file_sha256="score-sha",
        metadata={"contract_fingerprint": f"contract-{task}"},
        value=score,
    )
    carriage = _carriage()
    carriage_artifact = ReadOnlyCacheArtifact(
        path=tmp_path / task / "fields.pt",
        file_sha256="carriage-sha",
        metadata={"contract_fingerprint": f"carriage-{task}"},
        value=carriage,
    )
    return CachedModel(
        task=task,
        artifact_task=task,
        root=tmp_path,
        score_artifact=score_artifact,
        score=score,
        model_record={
            "train_seed": 42,
            "test_metric": 0.08,
            "validation_metric": 0.09,
            "parameter_count": 1000,
        },
        carriage_artifact=carriage_artifact,
        carriage=carriage,
    )


def test_grouped_head_profile_reconstructs_and_normalizes():
    score = _score()
    labels, grouped = group_distance(
        score["channels"]["semantic"]["heatmap_exact_head"], score["axis"]
    )
    assert labels == ("0", "1", "2", "3", "4-7", "8+", "virtual")
    np.testing.assert_allclose(
        grouped.sum(axis=-1), score["channels"]["semantic"]["raw"]
    )
    _, equal_head, activity_weighted = score_profile(score, "semantic")
    np.testing.assert_allclose(equal_head.sum(), 1.0)
    np.testing.assert_allclose(activity_weighted.sum(), 1.0)


def test_inventory_recognizes_historical_local_rrwp_task_name(tmp_path):
    task_dir = tmp_path / "zinc_1hop_local" / "seed_42"
    (task_dir / "cache/scores").mkdir(parents=True)
    (task_dir / "cache/scores/raw.pt").touch()
    (task_dir / "model.json").touch()
    inventory = cache_inventory(
        [tmp_path], tasks=("zinc_1hop_localrrwp",), train_seed=42
    )
    assert inventory[0]["complete_score_locations"] == 1
    assert inventory[0]["matches"][0]["artifact_task"] == "zinc_1hop_local"


def test_cached_interval_width_and_carriage_are_normalized():
    labels, relative_width, reportable = score_interval_width_profile(
        _score(), "semantic"
    )
    assert labels[-1] == "virtual"
    assert reportable.all()
    assert np.isfinite(relative_width).all()

    labels, profile, total, events = carriage_profile(_carriage(), "semantic")
    assert labels[:2] == ("0", "1")
    np.testing.assert_allclose(profile[:2], (0.375, 0.625))
    np.testing.assert_allclose(profile.sum(), 1.0)
    assert total == 4.0
    assert events == 2


def test_fast_run_writes_tables_and_aligned_figures(tmp_path, monkeypatch):
    models = [
        _model(tmp_path, "zinc_1hop_localrrwp", include_virtual=False),
        _model(tmp_path, "zinc_1hop", include_virtual=True),
    ]
    monkeypatch.setattr(
        "graph_specialisation_metrics.zinc_cached_rrwp_comparison.load_cached_models",
        lambda *args, **kwargs: models,
    )
    output = tmp_path / "output"
    result = run(
        [tmp_path],
        output,
        tasks=("zinc_1hop_localrrwp", "zinc_1hop"),
        train_seed=42,
    )
    assert (output / "model_summary.csv").is_file()
    assert (output / "head_scores.csv").is_file()
    assert (output / "head_score_distance.csv").is_file()
    assert (output / "pairwise_comparisons.csv").is_file()
    assert (output / "summary.json").is_file()
    assert len(result["pairwise_comparisons"]) == 1
    assert np.isfinite(
        result["pairwise_comparisons"][0]["semantic_score_profile_total_variation"]
    )
    assert len(result["figures"]) == 4
    assert all(Path(path).is_file() for path in result["figures"])
