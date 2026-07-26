from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from graph_specialisation_metrics.methodology.cache import (
    ReadOnlyCacheArtifact,
    StaleCacheError,
    load_cache_artifact_file,
)
from graph_specialisation_metrics.methodology.protocol import (
    PROTOCOL_VERSION,
    stable_hash,
)
from graph_specialisation_metrics.synthetic.nar_causal_transition import (
    _expected_provenance,
    _provenance_path,
    load_transition_causal_artifact,
)
from graph_specialisation_metrics.synthetic.nar_methodology_paper_v3 import (
    build_parser,
    causal_engagement_rows,
    causal_engagement_transition_rows,
    discriminant_rows,
    interaction_outlier_rows,
    localisation_rows,
    plot_compact_causal_validation,
    plot_complete_grounding,
    plot_causal_engagement_and_capacity,
    plot_conditional_fingerprint_supplement,
    plot_discriminant_validity,
    plot_family_overlap,
    plot_localisation_and_counterfactual,
    plot_organisation,
    plot_specificity_vs_engagement,
)
from graph_specialisation_metrics.synthetic.nar_methodology_paper import PaperInputs


def test_localisation_uses_matching_core_channel_and_is_bounded():
    rows = []
    values = {
        "semantic_query": 3.0,
        "semantic_record": 1.0,
        "structural_query": 2.0,
        "structural_record": 6.0,
    }
    for family in ("semantic_leaning", "structural_leaning"):
        for component, estimate in values.items():
            rows.append(
                {
                    "model": "dense",
                    "N": 16,
                    "seed": 0,
                    "family": family,
                    "component": component,
                    "estimate": estimate,
                }
            )

    output = {
        row["family"]: row["query_localisation"]
        for row in localisation_rows(rows)
    }
    assert output["semantic_leaning"] == pytest.approx(0.5)
    assert output["structural_leaning"] == pytest.approx(-0.5)
    assert all(-1 <= value <= 1 for value in output.values())


def test_discriminant_gap_is_same_minus_cross_channel():
    rows = []
    values = {
        "S_semantic_vs_G_semantic": 0.7,
        "S_semantic_vs_G_structural_control": 0.4,
        "S_structural_vs_G_structural": 0.8,
        "S_structural_vs_G_semantic_control": -0.1,
    }
    for test, estimate in values.items():
        rows.append(
            {
                "model": "2hop",
                "N": 32,
                "seed": 1,
                "section": "raw_score_calibration",
                "test": test,
                "estimate": estimate,
            }
        )
    output = {
        row["channel"]: row["specificity_gap"]
        for row in discriminant_rows(rows)
    }
    assert output["semantic"] == pytest.approx(0.3)
    assert output["structural"] == pytest.approx(0.9)


def test_family_interaction_outlier_audit_flags_single_seed_excursion():
    rows = [
        {
            "model": "2hop",
            "N": 16,
            "seed": seed,
            "section": "family_interaction",
            "test": "gross_family",
            "kind": "core_families",
            "estimate": value,
        }
        for seed, value in enumerate((0.4, 0.5, 125.0))
    ]
    audit = interaction_outlier_rows(rows)

    assert audit[2]["flag_abs_robust_z_gt_3_5"] is True
    assert audit[0]["headline_status"] == "demoted_pending_outlier_audit"


def test_v3_defaults_require_complete_score_and_causal_capacity_grid():
    args = build_parser().parse_args([])

    assert args.paper_analysis_name == "nar_methodology_paper_v3"
    assert args.score_ns == "4,8,16,32,64"
    assert args.canonical_causal_ns == "4,16,64"
    assert args.transition_causal_ns == "8,32"
    assert args.performance_ns.endswith(",80")


def test_absolute_causal_engagement_retention_does_not_use_normalised_J():
    models = ("dense",)
    seeds = (0,)
    ns = (4, 8)
    score_bindings = {}
    causal = {}
    performance = []
    for records, scale, accuracy in ((4, 2.0, 1.0), (8, 1.0, 0.5)):
        coordinates = SimpleNamespace(
            semantic_mean=4.0 * scale,
            structural_mean=2.0 * scale,
            joint_sensitivity=np.asarray([[0.5, 1.5]]),
            selectivity=np.asarray([[-0.4, 0.6]]),
            active=np.asarray([[True, True]]),
        )
        score_bindings[("dense", records, 0)] = SimpleNamespace(
            score_artifact=SimpleNamespace(value={"coordinates": coordinates})
        )
        causal[("dense", records, 0)] = {
            "summary": {
                "gross_reference_scales": {
                    "semantic": scale,
                    "structural": scale,
                },
                "necessity_reference_scales": {
                    "semantic": 2.0 * scale,
                    "structural": 0.5 * scale,
                },
            },
            "clean_ablation": {
                "head_L0_H0": {
                    "prediction_movement": scale,
                    "registered_metric_clean": accuracy,
                    "registered_metric_ablated": accuracy - 0.1,
                },
                "head_L0_H1": {
                    "prediction_movement": 2.0 * scale,
                    "registered_metric_clean": accuracy,
                    "registered_metric_ablated": accuracy - 0.2,
                },
            },
        }
        performance.append(
            {"model": "dense", "N": records, "seed": 0, "accuracy": accuracy}
        )
    inputs = PaperInputs(
        score_bindings=score_bindings,
        causal=causal,
        role_results={},
        counterfactual={},
        carriage={},
        best_seeds={},
        performance=performance,
    )

    rows = causal_engagement_rows(
        inputs,
        models=models,
        seeds=seeds,
        ns=ns,
    )
    by_n = {int(row["N"]): row for row in rows}

    assert by_n[4]["gross_joint_geomean_retention"] == pytest.approx(1.0)
    assert by_n[8]["gross_joint_geomean_retention"] == pytest.approx(0.5)
    assert by_n[8]["gross_joint_geomean_log2_retention"] == pytest.approx(-1.0)
    assert by_n[8]["raw_joint_geomean_log2_retention"] == pytest.approx(-1.0)
    assert "J_retention" not in by_n[8]
    assert "mean_h(J)=1" in by_n[8]["J_cross_N_status"]

    transitions, trajectories = causal_engagement_transition_rows(
        rows,
        models=models,
        seeds=seeds,
        ns=ns,
    )
    assert transitions[0]["delta_log2_raw_score_engagement"] == pytest.approx(-1.0)
    assert transitions[0]["delta_log2_gross_engagement"] == pytest.approx(-1.0)
    assert transitions[0]["delta_chance_adjusted_accuracy"] < 0
    assert len(trajectories["raw_score"]["dense:seed0"]) == 1
    assert len(trajectories["gross"]["dense:seed0"]) == 1


def test_transition_causal_loader_is_bound_to_source_score_hash(tmp_path):
    score_path = tmp_path / "score.pt"
    score_payload = {
        "metadata": {
            "protocol_version": PROTOCOL_VERSION,
            "contract_fingerprint": "score-contract",
            "contract": {},
        },
        "value": {"families": {"semantic_leaning": ((0, 0),)}},
    }
    torch.save(score_payload, score_path)
    score_artifact = ReadOnlyCacheArtifact(
        path=score_path,
        file_sha256="score-sha",
        metadata=score_payload["metadata"],
        value=score_payload["value"],
    )
    binding = SimpleNamespace(
        task="nar_dense_N8",
        seed=0,
        checkpoint_sha256="checkpoint-sha",
        source_contract_fingerprint="score-contract",
        score_artifact=score_artifact,
    )
    causal_path = (
        tmp_path
        / "canonical"
        / binding.task
        / "seed_0"
        / "cache"
        / "causal"
        / "validation.pt"
    )
    causal_path.parent.mkdir(parents=True)
    causal_contract = {
        "task": binding.task,
        "train_seed": 0,
        "checkpoint_sha256": "checkpoint-sha",
    }
    torch.save(
        {
            "metadata": {
                "protocol_version": PROTOCOL_VERSION,
                "contract_fingerprint": stable_hash(causal_contract),
                "contract": causal_contract,
            },
            "value": {"families": score_payload["value"]["families"]},
        },
        causal_path,
    )
    causal_artifact = load_cache_artifact_file(causal_path)
    provenance = {
        **_expected_provenance(binding),
        "causal_artifact_path": str(causal_path),
        "causal_artifact_sha256": causal_artifact.file_sha256,
    }
    provenance_path = _provenance_path(tmp_path, binding.task, 0)
    provenance_path.parent.mkdir(parents=True)
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    loaded = load_transition_causal_artifact(
        extension_root=tmp_path,
        binding=binding,
    )
    assert loaded["families"] == score_payload["value"]["families"]

    changed_binding = SimpleNamespace(**binding.__dict__)
    changed_binding.score_artifact = ReadOnlyCacheArtifact(
        path=score_path,
        file_sha256="different-score-sha",
        metadata=score_payload["metadata"],
        value=score_payload["value"],
    )
    with pytest.raises(StaleCacheError, match="source_score_sha256"):
        load_transition_causal_artifact(
            extension_root=tmp_path,
            binding=changed_binding,
        )


def test_v3_publication_figures_render_without_overlapping_legacy_panels(tmp_path):
    pytest.importorskip("matplotlib")
    models = ("1hop", "2hop", "dense")
    seeds = (0, 1, 2)
    ns = (4, 8)
    performance = [
        {"model": model, "N": records, "seed": seed, "accuracy": 0.9}
        for model in models
        for records in ns
        for seed in seeds
    ]
    inputs = PaperInputs(
        score_bindings={},
        causal={},
        role_results={},
        counterfactual={},
        carriage={},
        best_seeds={},
        performance=performance,
    )
    raw_tests = (
        ("S_semantic_vs_G_semantic", r"$S_{\rm sem}\leftrightarrow G_{\rm sem}$"),
        ("S_structural_vs_G_structural", r"$S_{\rm str}\leftrightarrow G_{\rm str}$"),
        (
            "S_semantic_vs_G_structural_control",
            r"$S_{\rm sem}\leftrightarrow G_{\rm str}$",
        ),
        (
            "S_structural_vs_G_semantic_control",
            r"$S_{\rm str}\leftrightarrow G_{\rm sem}$",
        ),
    )
    coordinate_tests = (
        ("J_vs_clean_prediction_movement", r"$J\leftrightarrow$ clean ablation"),
        ("J_vs_gross_total", r"$J\leftrightarrow$ total gross"),
        ("J_vs_necessity_total", r"$J\leftrightarrow$ total necessity"),
        ("D_rel_vs_gross_contrast", r"$D_{\rm rel}\leftrightarrow$ gross contrast"),
        (
            "D_rel_vs_necessity_contrast",
            r"$D_{\rm rel}\leftrightarrow$ necessity contrast",
        ),
    )
    causal_rows = []
    for model in models:
        for records in ns:
            for seed in seeds:
                for test, label in raw_tests:
                    causal_rows.append(
                        {
                            "model": model,
                            "N": records,
                            "seed": seed,
                            "section": "raw_score_calibration",
                            "test": test,
                            "label": label,
                            "estimate": 0.5,
                        }
                    )
                for test, label in coordinate_tests:
                    causal_rows.append(
                        {
                            "model": model,
                            "N": records,
                            "seed": seed,
                            "section": "coordinate_validation",
                            "test": test,
                            "label": label,
                            "estimate": 0.6,
                        }
                    )
    localisation = [
        {
            "model": model,
            "N": records,
            "seed": seed,
            "family": family,
            "query_localisation": 0.3 if family == "structural_leaning" else -0.2,
        }
        for model in models
        for records in ns
        for seed in seeds
        for family in ("semantic_leaning", "structural_leaning")
    ]
    conditional = [
        {
            "model": model,
            "N": 4,
            "seed": seed,
            "family": family,
            "component": component,
            "estimate": 1.0,
        }
        for model in models
        for seed in seeds
        for family in ("semantic_leaning", "structural_leaning")
        for component in (
            "semantic_query",
            "semantic_record",
            "structural_query",
            "structural_record",
        )
    ]
    counterfactual = [
        {
            "model": model,
            "N": records,
            "analysis_set": "successful_exact_recall",
            "estimand": "adjusted",
            "estimate": 0.3,
            "ci95_low": 0.1,
            "ci95_high": 0.5,
        }
        for model in models
        for records in (*ns, "pooled")
    ]
    organisation = [
        {
            "model": model,
            "N": records,
            "seed": seed,
            "active_fraction": 0.5,
            "top20_J_share": 0.6,
            "median_active_abs_D_rel": 0.3,
            "semantic_family_heads": 2,
            "structural_family_heads": 2,
        }
        for model in models
        for records in ns
        for seed in seeds
    ]
    overlaps = [
        {
            "model": model,
            "N": records,
            "seed": seed,
            "core_family": core,
            "role_family": role,
            "jaccard": 0.25,
        }
        for model in models
        for records in ns
        for seed in seeds
        for core in ("semantic_leaning", "structural_leaning")
        for role in ("address", "content")
    ]
    discriminant = [
        {
            "model": model,
            "N": records,
            "seed": seed,
            "channel": channel,
            "specificity_gap": 0.2,
        }
        for model in models
        for records in ns
        for seed in seeds
        for channel in ("semantic", "structural")
    ]
    engagement = [
        {
            "model": model,
            "N": records,
            "seed": seed,
            "chance_adjusted_accuracy": 0.9 - 0.1 * ns.index(records),
            "raw_semantic_mean_log2_retention": -0.45 * ns.index(records),
            "raw_structural_mean_log2_retention": -0.55 * ns.index(records),
            "gross_joint_geomean_log2_retention": -0.5 * ns.index(records),
            "gross_semantic_log2_retention": -0.4 * ns.index(records),
            "gross_structural_log2_retention": -0.6 * ns.index(records),
            "necessity_joint_geomean_log2_retention": -0.4 * ns.index(records),
            "median_active_abs_D_rel": 0.2 + 0.1 * ns.index(records),
        }
        for model in models
        for records in ns
        for seed in seeds
    ]
    engagement_transitions = [
        {
            "model": model,
            "seed": seed,
            "transition": "4->8",
            "delta_log2_gross_engagement": -0.5,
            "delta_chance_adjusted_accuracy": -0.1,
        }
        for model in models
        for seed in seeds
    ]
    engagement_statistics = [
        {
            "endpoint": "gross",
            "spearman_rho": 0.8,
            "ci95_low": 0.5,
            "ci95_high": 0.95,
        }
    ]

    plot_compact_causal_validation(
        inputs,
        output_dir=tmp_path,
        models=models,
        seeds=seeds,
        records=4,
        rows=causal_rows,
    )
    plot_complete_grounding(
        inputs,
        output_dir=tmp_path,
        models=models,
        performance_ns=ns,
        causal_ns=ns,
        seeds=seeds,
        causal_rows=causal_rows,
    )
    plot_localisation_and_counterfactual(
        inputs,
        output_dir=tmp_path,
        models=models,
        seeds=seeds,
        score_ns=ns,
        counterfactual_rows=counterfactual,
        localisation=localisation,
    )
    plot_conditional_fingerprint_supplement(
        inputs,
        output_dir=tmp_path,
        models=models,
        seeds=seeds,
        records=4,
        rows=conditional,
    )
    plot_organisation(
        organisation,
        output_dir=tmp_path,
        models=models,
        seeds=seeds,
        ns=ns,
    )
    plot_family_overlap(
        overlaps,
        output_dir=tmp_path,
        models=models,
        ns=ns,
    )
    plot_discriminant_validity(
        discriminant,
        output_dir=tmp_path,
        models=models,
        seeds=seeds,
        ns=ns,
    )
    plot_causal_engagement_and_capacity(
        engagement,
        engagement_transitions,
        engagement_statistics,
        output_dir=tmp_path,
        models=models,
        seeds=seeds,
        ns=ns,
    )
    plot_specificity_vs_engagement(
        engagement,
        output_dir=tmp_path,
        models=models,
        seeds=seeds,
        ns=ns,
    )

    assert len(list(tmp_path.rglob("*.png"))) == 9
