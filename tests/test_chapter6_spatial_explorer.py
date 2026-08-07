import json
from pathlib import Path

import matplotlib
import numpy as np
import pytest
import torch

from graph_specialisation_metrics.chapter6_spatial_explorer import (
    graph_spatial_metrics,
    head_role_score_allocation,
    head_metrics,
    layer_distance_profiles,
    layer_score_organisation,
    layer_summary,
    load_models,
    molecular_scale_relationships,
    reach_mismatch_summary,
    representative_reach_mismatches,
    run,
    score_organisation_similarity,
    spatial_width_bootstrap,
    vnode_cross_layer_relationships,
    width_by_head_role,
    width_contribution_profiles,
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
            "graph_distance_support": {
                10: np.ones(len(axis)),
                11: np.ones(len(axis)),
            },
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
            "raw_semantic": raw_semantic,
            "raw_structural": raw_structural,
            "normalized_semantic": raw_semantic / raw_semantic.mean(),
            "normalized_structural": raw_structural / raw_structural.mean(),
            "joint_sensitivity": joint,
            "selectivity": (raw_semantic - raw_structural) / joint,
        },
        "families": {
            "semantic_leaning": ((0, 0),),
            "structural_leaning": ((1, 0),),
            "central_responsive": ((0, 1),),
            "inactive": (),
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
    assert first["semantic_attention_reach_gap"] == pytest.approx(-1.0)
    assert first["structural_attention_reach_gap"] == pytest.approx(0.0)
    assert first["max_abs_attention_reach_gap"] == pytest.approx(1.0)
    assert first["dominant_reach_gap_channel"] == "semantic"
    assert first["overlap"] == pytest.approx(0.0)
    assert np.isfinite(first["semantic_expected_distance_sem"])
    assert first["family"] == "semantic_leaning"
    assert first["raw_semantic_score"] == pytest.approx(1.0)
    assert first["semantic_opportunity_expected_distance"] == pytest.approx(0.0)

    summary = layer_summary(rows)
    sources = {row["source"] for row in summary}
    assert sources == {
        "semantic",
        "structural",
        "semantic_opportunity",
        "structural_opportunity",
        "attention",
    }
    semantic_layer_zero = next(
        row for row in summary if row["source"] == "semantic" and row["layer"] == 0
    )
    assert np.isfinite(semantic_layer_zero["spatial_variance_q1"])
    assert np.isfinite(semantic_layer_zero["spatial_variance_q3"])

    distance_rows = layer_distance_profiles(models)
    assert {row["profile_kind"] for row in distance_rows} == {
        "score_mass",
        "per_opportunity",
    }
    bootstrap_rows = spatial_width_bootstrap(models, replicates=200, seed=0)
    assert len(bootstrap_rows) == 2
    assert bootstrap_rows[0]["graphs"] == 2
    contribution_rows = width_contribution_profiles(models)
    assert {row["weighting"] for row in contribution_rows} == {
        "equal_head",
        "J_weighted",
    }
    role_rows = width_by_head_role(rows)
    assert any(row["family"] == "all_active" for row in role_rows)
    graph_rows = graph_spatial_metrics(models)
    assert all(row["num_nodes"] == pytest.approx(3.0) for row in graph_rows)
    assert all(row["diameter"] == pytest.approx(2.0) for row in graph_rows)
    organisation_rows = layer_score_organisation(rows)
    assert sum(row["joint_sensitivity_share"] for row in organisation_rows) == pytest.approx(
        1.0
    )
    role_allocation = head_role_score_allocation(models, rows, tail_distance=2.0)
    assert sum(row["joint_sensitivity_share"] for row in role_allocation) == pytest.approx(
        1.0
    )
    assert sum(
        row["structural_tail_share"]
        for row in role_allocation
        if np.isfinite(row["structural_tail_share"])
    ) == pytest.approx(1.0)
    similarity_rows = score_organisation_similarity(
        distance_rows, organisation_rows, ["zinc_1hop"]
    )
    assert similarity_rows[0]["model_wide_similarity"] == pytest.approx(1.0)
    assert similarity_rows[0]["layer_resolved_similarity"] == pytest.approx(1.0)
    mismatch_summary = reach_mismatch_summary(rows)
    assert mismatch_summary[0]["heads"] > 0
    mismatch_rows = representative_reach_mismatches(rows)
    assert len(mismatch_rows) == 1
    assert mismatch_rows[0]["role"] == "strongest active reach mismatch"


def test_loader_falls_back_to_equivalent_duplicate_carriage(tmp_path):
    preferred = tmp_path / "preferred"
    fallback = tmp_path / "fallback"
    _write_artifact(preferred, "zinc_1hop", with_carriage=False)
    _write_artifact(fallback, "zinc_1hop", with_carriage=True)
    preferred_score = preferred / "zinc_1hop/seed_42/cache/scores/raw.pt"
    payload = torch.load(preferred_score, map_location="cpu", weights_only=False)
    payload["metadata"]["protocol_version"] = "donor-swap-specialisation-carriage-v4"
    torch.save(payload, preferred_score)
    fallback_score = fallback / "zinc_1hop/seed_42/cache/scores/raw.pt"
    payload = torch.load(fallback_score, map_location="cpu", weights_only=False)
    payload["metadata"]["protocol_version"] = "donor-swap-specialisation-carriage-v3"
    torch.save(payload, fallback_score)

    models, warnings = load_models([preferred, fallback], ["zinc_1hop"], seed=42)
    assert models[0].score_path == preferred_score
    assert models[0].carriage_path == (fallback / "zinc_1hop/seed_42/cache/carriage/fields.pt")
    assert any("equivalent carriage artifact" in warning for warning in warnings)


def test_graphwise_followups_recover_cross_layer_and_scale_relationships():
    rows = []
    for graph in range(6):
        scale = float(graph + 3)
        rows.extend(
            [
                {
                    "task": "zinc_1hop_vnode",
                    "graph_id": graph,
                    "layer": 0,
                    "num_nodes": scale,
                    "diameter": scale,
                    "semantic_vnode_share": scale,
                    "structural_vnode_share": 0.0,
                    "semantic_expected_distance": 0.0,
                    "structural_expected_distance": 0.0,
                    "structural_excess_width": scale,
                },
                {
                    "task": "zinc_1hop_vnode",
                    "graph_id": graph,
                    "layer": 1,
                    "num_nodes": scale,
                    "diameter": scale,
                    "semantic_vnode_share": scale,
                    "structural_vnode_share": 0.0,
                    "semantic_expected_distance": scale,
                    "structural_expected_distance": scale,
                    "structural_excess_width": scale,
                },
            ]
        )
    cross_layer = vnode_cross_layer_relationships(rows)
    assert all(row["spearman_rho"] == pytest.approx(1.0) for row in cross_layer)
    scale_rows = molecular_scale_relationships(rows)
    width_rows = [row for row in scale_rows if row["outcome"] == "structural_excess_width"]
    assert width_rows
    assert all(row["spearman_rho"] == pytest.approx(1.0) for row in width_rows)


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
        "head_reach_mismatch_summary.csv",
        "representative_reach_mismatches.csv",
        "layer_spatial_summary.csv",
        "representative_heads.csv",
        "model_distance_profiles.csv",
        "score_profile_uncertainty.csv",
        "layer_distance_profiles.csv",
        "vnode_layer_allocation.csv",
        "spatial_width_graph_bootstrap.csv",
        "width_contributions_by_distance.csv",
        "width_by_head_role.csv",
        "layer_score_organisation.csv",
        "head_role_score_allocation.csv",
        "score_organisation_similarity.csv",
        "graph_spatial_metrics.csv",
        "vnode_cross_layer_relationships.csv",
        "molecular_scale_relationships.csv",
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
        "06_alignment_and_head_roles.png",
        "07_score_and_final_state_response.png",
        "08_raw_vs_opportunity_reach.png",
        "09_layer_distance_profiles_raw.png",
        "10_layer_distance_profiles_opportunity.png",
        "12_structural_minus_semantic_width.png",
        "13_width_excess_by_distance.png",
        "14_width_excess_by_head_role.png",
        "16_scale_and_structural_width.png",
        "18_layerwise_score_organisation.png",
        "19_head_role_score_allocation.png",
        "20_score_organisation_similarity.png",
        "21_attention_vs_score_reach.png",
        "22_head_attention_score_reach_gap.png",
        "23_head_score_landscapes.png",
    }


def test_colab_enables_two_reach_mismatch_examples_by_default():
    source = Path("experiments/zinc/analysis/chapter6_spatial_explorer_colab.py").read_text(
        encoding="utf-8"
    )
    assert "GENERATE_HEAD_CONTEXT = False" in source
    assert "GENERATE_REACH_MISMATCH_CONTEXT = True" in source
    assert "REACH_MISMATCH_CONTEXT_TASKS = TASKS" in source
    assert "HEAD_CONTEXT_GRAPH_INDICES = (0, 1)" in source
    assert 'context_name="reach_mismatch"' in source


def test_head_context_records_missing_representatives(tmp_path):
    from graph_specialisation_metrics.chapter6_head_context import (
        generate_head_context,
    )

    result = generate_head_context(
        [{"task": "zinc", "artifact_task": "zinc", "score": "/missing/raw.pt"}],
        [],
        tmp_path,
        tasks=("zinc",),
        graph_indices=(0, 1),
        context_name="reach_mismatch",
        verbose=False,
    )
    assert not result["outputs"]
    assert "no representative head rows" in result["warnings"][0]
    summary = json.loads(Path(result["summary_path"]).read_text(encoding="utf-8"))
    assert summary["context_name"] == "reach_mismatch"
    assert summary["graph_indices"] == [0, 1]
    assert summary["tasks"][0]["status"] == "skipped"


def test_head_context_uses_permissive_protocol_runtime():
    source = Path("src/graph_specialisation_metrics/chapter6_head_context.py").read_text(
        encoding="utf-8"
    )
    assert "require_protocol_match=False" in source
    assert "semantic_attention_reach_gap" in source
