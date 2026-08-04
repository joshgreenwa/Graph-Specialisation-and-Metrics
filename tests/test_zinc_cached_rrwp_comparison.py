import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from graph_specialisation_metrics.methodology.bootstrap import Interval
from graph_specialisation_metrics.methodology.cache import (
    ReadOnlyCacheArtifact,
    StaleCacheError,
)
from graph_specialisation_metrics.methodology.protocol import stable_hash
from graph_specialisation_metrics.zinc_cached_rrwp_comparison import (
    CachedModel,
    _resolve_task_root,
    cache_inventory,
    carriage_profile,
    group_distance,
    head_expected_distance_rows,
    head_profile_alignment_rows,
    head_profile_distance_decomposition_rows,
    head_spatial_width_rows,
    load_compatible_cache_artifact_file,
    load_or_compute_head_profile_alignment,
    run,
    score_interval_width_profile,
    score_profile,
    summarise_layerwise_distance_decomposition,
    summarise_layerwise_head_profile_alignment,
    summarise_layerwise_vnode_profiles,
    summarise_spatial_width_alignment,
    vnode_profile_rows,
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
            "heatmap_per_opportunity_head": exact,
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


def _model(
    tmp_path: Path,
    task: str,
    *,
    include_virtual: bool,
    include_carriage: bool = True,
) -> CachedModel:
    axis = (0, 1, 2, 3, 4, 8, "virtual") if include_virtual else (0, 1, 2, 3, 4, 8)
    score = _score(axis)
    score_artifact = ReadOnlyCacheArtifact(
        path=tmp_path / task / "raw.pt",
        file_sha256="score-sha",
        metadata={"contract_fingerprint": f"contract-{task}"},
        value=score,
    )
    carriage = _carriage() if include_carriage else None
    carriage_artifact = (
        ReadOnlyCacheArtifact(
            path=tmp_path / task / "fields.pt",
            file_sha256="carriage-sha",
            metadata={"contract_fingerprint": f"carriage-{task}"},
            value=carriage,
        )
        if carriage is not None
        else None
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


def test_legacy_v3_cache_is_read_only_and_fingerprint_validated(tmp_path):
    contract = {
        "task": "zinc_1hop",
        "checkpoint_sha256": "checkpoint",
        "repository_commit": "legacy-commit",
    }
    path = tmp_path / "raw.pt"
    payload = {
        "metadata": {
            "protocol_version": "donor-swap-specialisation-carriage-v3",
            "contract": contract,
            "contract_fingerprint": stable_hash(contract),
        },
        "value": {"sentinel": 7},
    }
    torch.save(payload, path)
    artifact = load_compatible_cache_artifact_file(path)
    assert artifact.value == {"sentinel": 7}
    assert artifact.metadata["protocol_version"].endswith("v3")

    payload["metadata"]["contract_fingerprint"] = "corrupt"
    torch.save(payload, path)
    with pytest.raises(StaleCacheError, match="fingerprint"):
        load_compatible_cache_artifact_file(path)


def test_root_resolution_prefers_v4_over_nonidentical_v3(tmp_path, monkeypatch):
    roots = [tmp_path / "v3_root", tmp_path / "v4_root"]
    protocols = (
        "donor-swap-specialisation-carriage-v3",
        "donor-swap-specialisation-carriage-v4",
    )
    artifacts = {}
    for root, protocol in zip(roots, protocols):
        task_dir = root / "zinc_1hop" / "seed_42"
        score = task_dir / "cache/scores/raw.pt"
        score.parent.mkdir(parents=True)
        score.touch()
        (task_dir / "model.json").touch()
        artifacts[score] = ReadOnlyCacheArtifact(
            path=score,
            file_sha256="sha",
            metadata={
                "protocol_version": protocol,
                "contract_fingerprint": protocol,
            },
            value={},
        )
    monkeypatch.setattr(
        "graph_specialisation_metrics.zinc_cached_rrwp_comparison."
        "load_compatible_score_artifact",
        lambda path, expected_task: artifacts[Path(path)],
    )
    root, artifact_task = _resolve_task_root(roots, "zinc_1hop", 42)
    assert root == roots[1]
    assert artifact_task == "zinc_1hop"


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


def test_layerwise_alignment_summarises_consistency_and_low_outliers(tmp_path):
    model = _model(tmp_path, "zinc_1hop", include_virtual=False)
    score = model.score
    width = len(score["axis"])
    semantic = np.zeros((2, 8, width), dtype=np.float64)
    structural = np.zeros_like(semantic)
    semantic[..., 0] = 1.0
    structural[..., 0] = 1.0
    structural[:, 7, 0] = 0.0
    structural[:, 7, 2] = 1.0
    for field in ("heatmap_exact_head", "heatmap_per_opportunity_head"):
        score["channels"]["semantic"][field] = semantic.copy()
        score["channels"]["structural"][field] = structural.copy()
    score["channels"]["semantic"]["raw"] = semantic.sum(axis=-1)
    score["channels"]["structural"]["raw"] = structural.sum(axis=-1)
    joint_sensitivity = np.ones((2, 8), dtype=np.float64)
    joint_sensitivity[:, 7] = 9.0
    score["coordinates"]["joint_sensitivity"] = joint_sensitivity
    score["coordinates"]["active"] = np.ones((2, 8), dtype=bool)
    score["coordinates"] = SimpleNamespace(**score["coordinates"])

    rows = head_profile_alignment_rows([model])
    assert len(rows) == 2 * 2 * 8
    layerwise, outliers = summarise_layerwise_head_profile_alignment(rows)
    mass_layer_zero = next(
        row
        for row in layerwise
        if row["profile_kind"] == "score_mass" and row["layer"] == 0
    )
    assert mass_layer_zero["overlap_median"] == pytest.approx(1.0)
    assert mass_layer_zero["overlap_iqr"] == pytest.approx(0.0)
    assert mass_layer_zero["overlap_min_head"] == 7
    assert mass_layer_zero["low_overlap_outlier_heads"] == "7"
    assert mass_layer_zero["low_overlap_outlier_count"] == 1
    assert mass_layer_zero["overlap_mean"] == pytest.approx(7.0 / 8.0)
    assert mass_layer_zero["activity_weighted_overlap"] == pytest.approx(7.0 / 16.0)
    assert mass_layer_zero["activity_weighted_minus_median"] == pytest.approx(
        -9.0 / 16.0
    )
    assert mass_layer_zero[
        "joint_sensitivity_share_overlap_below_0_7"
    ] == pytest.approx(9.0 / 16.0)
    expected_rows = head_expected_distance_rows(rows)
    displaced_expected = next(
        row
        for row in expected_rows
        if row["profile_kind"] == "score_mass"
        and row["layer"] == 0
        and row["head"] == 7
    )
    assert displaced_expected["semantic_expected_molecular_distance"] == pytest.approx(
        0.0
    )
    assert displaced_expected[
        "structural_expected_molecular_distance"
    ] == pytest.approx(2.0)
    assert displaced_expected[
        "structural_minus_semantic_expected_distance"
    ] == pytest.approx(2.0)
    assert {
        (row["profile_kind"], row["layer"], row["head"])
        for row in outliers
    } == {
        ("score_mass", 0, 7),
        ("score_mass", 1, 7),
        ("per_opportunity", 0, 7),
        ("per_opportunity", 1, 7),
    }

    distance_rows = head_profile_distance_decomposition_rows([model])
    displaced = [
        row
        for row in distance_rows
        if row["profile_kind"] == "score_mass"
        and row["layer"] == 0
        and row["head"] == 7
    ]
    assert sum(float(row["tv_contribution"]) for row in displaced) == pytest.approx(1.0)
    assert {
        row["distance"]: row["structural_minus_semantic"] for row in displaced
    } == {"0": -1.0, "1": 0.0, "2": 1.0, "3": 0.0, "4": 0.0, "8": 0.0}
    distance_layerwise = summarise_layerwise_distance_decomposition(distance_rows)
    assert any(
        row["profile_kind"] == "score_mass"
        and row["layer"] == 0
        and row["distance_group"] == "2"
        and row["tv_contribution_mean"] == pytest.approx(0.5 / 8.0)
        for row in distance_layerwise
    )


def test_vnode_diagnostic_removes_and_renormalizes_virtual_bin(tmp_path):
    model = _model(tmp_path, "zinc_1hop_vnode", include_virtual=True)
    width = len(model.score["axis"])
    semantic = np.zeros((2, 2, width), dtype=np.float64)
    structural = np.zeros_like(semantic)
    semantic[..., 0] = 1.0
    semantic[..., -1] = 1.0
    structural[..., 0] = 1.0
    structural[..., -1] = 3.0
    for field in ("heatmap_exact_head", "heatmap_per_opportunity_head"):
        model.score["channels"]["semantic"][field] = semantic.copy()
        model.score["channels"]["structural"][field] = structural.copy()
    model.score["channels"]["semantic"]["raw"] = semantic.sum(axis=-1)
    model.score["channels"]["structural"]["raw"] = structural.sum(axis=-1)

    rows = vnode_profile_rows([model])
    assert len(rows) == 2 * 2 * 2
    first = rows[0]
    assert first["full_overlap"] == pytest.approx(0.75)
    assert first["molecular_only_overlap"] == pytest.approx(1.0)
    assert first["semantic_virtual_share"] == pytest.approx(0.5)
    assert first["structural_virtual_share"] == pytest.approx(0.75)
    assert first["virtual_tv_contribution"] == pytest.approx(0.125)
    layerwise = summarise_layerwise_vnode_profiles(rows)
    assert len(layerwise) == 2 * 2
    assert layerwise[0]["molecular_minus_full_overlap_median"] == pytest.approx(0.25)


def test_spatial_width_distinguishes_broad_profiles_with_the_same_centroid(tmp_path):
    model = _model(tmp_path, "zinc_1hop", include_virtual=False)
    width = len(model.score["axis"])
    semantic = np.zeros((2, 2, width), dtype=np.float64)
    structural = np.zeros_like(semantic)
    semantic[..., 1] = 1.0
    structural[..., 1] = 1.0
    semantic[0, 0] = 0.0
    semantic[0, 0, 0] = 0.5
    semantic[0, 0, 2] = 0.5
    for field in ("heatmap_exact_head", "heatmap_per_opportunity_head"):
        model.score["channels"]["semantic"][field] = semantic.copy()
        model.score["channels"]["structural"][field] = structural.copy()

    alignment_rows = head_profile_alignment_rows([model])
    width_rows = head_spatial_width_rows(alignment_rows)
    broad = next(
        row
        for row in width_rows
        if row["profile_kind"] == "score_mass"
        and row["layer"] == 0
        and row["head"] == 0
    )
    alignment = next(
        row
        for row in alignment_rows
        if row["profile_kind"] == "score_mass"
        and row["layer"] == 0
        and row["head"] == 0
    )
    assert alignment["semantic_centroid"] == pytest.approx(1.0)
    assert alignment["structural_centroid"] == pytest.approx(1.0)
    assert broad["semantic_spatial_variance"] == pytest.approx(1.0)
    assert broad["structural_spatial_variance"] == pytest.approx(0.0)
    assert broad["semantic_spatial_entropy_nats"] == pytest.approx(np.log(2.0))
    assert broad["structural_spatial_entropy_nats"] == pytest.approx(0.0)
    assert broad["semantic_effective_distance_bins"] == pytest.approx(2.0)


def test_spatial_width_alignment_reports_rank_and_identity_agreement():
    rows = [
        {
            "task": "zinc_1hop",
            "profile_kind": "score_mass",
            "semantic_spatial_variance": value,
            "structural_spatial_variance": 2.0 * value,
            "semantic_spatial_entropy_nats": value,
            "structural_spatial_entropy_nats": 2.0 * value,
        }
        for value in (1.0, 2.0, 3.0)
    ]
    summaries = summarise_spatial_width_alignment(rows)
    variance = next(row for row in summaries if row["metric"] == "spatial_variance")
    assert variance["pearson_correlation"] == pytest.approx(1.0)
    assert variance["spearman_correlation"] == pytest.approx(1.0)
    assert variance["concordance_correlation"] < 1.0
    assert variance["structural_minus_semantic_median"] == pytest.approx(2.0)


def test_run_identifies_the_model_that_failed_to_load(tmp_path, monkeypatch):
    local = _model(tmp_path, "zinc_1hop_localrrwp", include_virtual=False)

    def load_one(*args, **kwargs):
        task = kwargs["tasks"][0]
        if task == "zinc_1hop":
            raise ValueError("corrupt score fixture")
        return [local]

    monkeypatch.setattr(
        "graph_specialisation_metrics.zinc_cached_rrwp_comparison.load_cached_models",
        load_one,
    )
    with pytest.raises(
        RuntimeError,
        match="while loading zinc_1hop: ValueError: corrupt score fixture",
    ):
        run(
            [tmp_path],
            tmp_path / "output",
            tasks=("zinc_1hop_localrrwp", "zinc_1hop"),
            verbose=False,
        )


def test_fast_run_writes_tables_and_aligned_figures(tmp_path, monkeypatch):
    models = [
        _model(
            tmp_path,
            "zinc_1hop_localrrwp",
            include_virtual=False,
            include_carriage=False,
        ),
        _model(tmp_path, "zinc_1hop", include_virtual=True),
    ]
    monkeypatch.setattr(
        "graph_specialisation_metrics.zinc_cached_rrwp_comparison.load_cached_models",
        lambda *args, **kwargs: [
            model for model in models if model.task in kwargs["tasks"]
        ],
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
    assert (output / "head_profile_alignment.csv").is_file()
    assert (output / "head_profile_alignment_summary.csv").is_file()
    assert (output / "head_profile_alignment_layerwise.csv").is_file()
    assert (output / "head_profile_alignment_outliers.csv").is_file()
    assert (output / "head_profile_distance_decomposition.csv").is_file()
    assert (output / "head_profile_distance_decomposition_layerwise.csv").is_file()
    assert (output / "vnode_profile_alignment.csv").is_file()
    assert (output / "vnode_profile_alignment_layerwise.csv").is_file()
    assert (output / "head_profile_activity_weighted_overlap.csv").is_file()
    assert (output / "head_expected_score_distance.csv").is_file()
    assert (output / "head_score_spatial_width.csv").is_file()
    assert (output / "head_score_spatial_width_alignment.csv").is_file()
    assert not (output / "head_profile_alignment_permutation.csv").exists()
    assert (output / "cache/head_profile_alignment.json").is_file()
    assert (output / "summary.json").is_file()
    with (output / "model_summary.csv").open(encoding="utf-8", newline="") as handle:
        summary_rows = list(csv.DictReader(handle))
    assert len(summary_rows) == 2
    assert summary_rows[0]["semantic_carriage_total"] == ""
    assert float(summary_rows[1]["semantic_carriage_total"]) == pytest.approx(4.0)
    assert len(result["pairwise_comparisons"]) == 1
    assert np.isfinite(
        result["pairwise_comparisons"][0]["semantic_score_profile_total_variation"]
    )
    assert result["head_profile_alignment"]["cache_status"] == "miss"
    _, cache_path, cache_status = load_or_compute_head_profile_alignment(
        models, output
    )
    assert cache_path == output / "cache/head_profile_alignment.json"
    assert cache_status == "hit"
    assert len(result["figures"]) == 28
    assert all(Path(path).is_file() for path in result["figures"])
