from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from graph_specialisation_metrics.methodology.protocol import BootstrapPolicy
from graph_specialisation_metrics.synthetic.nar_methodology_paper import (
    PaperInputs,
    _counterfactual_vector_transform,
    build_parser,
    causal_summary_rows,
    conditional_source_scores,
    counterfactual_graph_observations,
)


def _event(graph: int, source: int, draw: int, score) -> dict[str, object]:
    return {
        "graph_id": graph,
        "source": source,
        "draw": draw,
        "score": np.asarray(score, dtype=np.float64),
    }


def test_conditional_source_scores_keep_unconditional_channel_scale():
    semantic_events = []
    structural_events = []
    for graph in (0, 1):
        for draw in (0, 1):
            semantic_events.extend(
                (
                    _event(graph, 2, draw, [[2.0, 4.0]]),
                    _event(graph, 3, draw, [[6.0, 8.0]]),
                )
            )
            structural_events.extend(
                (
                    _event(graph, 2, draw, [[1.0, 3.0]]),
                    _event(graph, 3, draw, [[5.0, 7.0]]),
                )
            )
    scores = {
        "channels": {
            "semantic": {
                "raw": np.asarray([[4.0, 6.0]]),
                "events": semantic_events,
            },
            "structural": {
                "raw": np.asarray([[3.0, 5.0]]),
                "events": structural_events,
            },
        }
    }

    conditional = conditional_source_scores(scores)

    np.testing.assert_allclose(conditional["semantic_query"], [[0.4, 0.8]])
    np.testing.assert_allclose(conditional["semantic_record"], [[1.2, 1.6]])
    np.testing.assert_allclose(conditional["structural_query"], [[0.25, 0.75]])
    np.testing.assert_allclose(conditional["structural_record"], [[1.25, 1.75]])
    assert float(conditional["semantic_reconstruction_error"]) == pytest.approx(0.0)
    assert float(conditional["structural_reconstruction_error"]) == pytest.approx(0.0)
    # Separate role renormalisation would force both role means to one; the registered conditional
    # analysis deliberately preserves the amplitude difference instead.
    assert np.mean(conditional["semantic_query"]) == pytest.approx(0.6)
    assert np.mean(conditional["semantic_record"]) == pytest.approx(1.4)


def _counterfactual_row(
    graph: int,
    role: str,
    family: str,
    mediation: float,
    *,
    correct: bool = True,
) -> dict[str, object]:
    return {
        "graph_id": graph,
        "role": role,
        "family": family,
        "draw": 0,
        "estimable": True,
        "clean_correct": correct,
        "counterfactual_correct": correct,
        "directional_numerator": mediation * 2.0,
        "full_margin_change": 2.0,
    }


def _counterfactual_result() -> dict[str, object]:
    rows = []
    effects = {
        ("query", "address"): 0.8,
        ("query", "content"): 0.2,
        ("value", "address"): 0.1,
        ("value", "content"): 0.7,
        ("query", "address_control"): 0.3,
        ("query", "content_control"): 0.3,
        ("value", "address_control"): 0.3,
        ("value", "content_control"): 0.3,
    }
    for graph in (0, 1):
        for (role, family), effect in effects.items():
            rows.append(
                _counterfactual_row(
                    graph,
                    role,
                    family,
                    effect,
                    correct=graph == 0,
                )
            )
    return {"primary": True, "rows": rows}


def test_counterfactual_summary_uses_paired_control_adjusted_double_difference():
    results = {("dense", 16, 0): _counterfactual_result()}
    observations, support = counterfactual_graph_observations(
        results,
        model="dense",
        records=16,
        require_correct=True,
    )

    assert len(observations) == 1
    assert support["graphs"] == 1
    family, control, adjusted = _counterfactual_vector_transform(
        observations[0].value
    )
    assert family == pytest.approx(1.2)
    assert control == pytest.approx(0.0)
    assert adjusted == pytest.approx(1.2)

    all_observations, _ = counterfactual_graph_observations(
        results,
        model="dense",
        records=16,
        require_correct=False,
    )
    assert len(all_observations) == 2


def test_causal_summary_keeps_J_and_D_rel_attached_to_the_right_endpoints():
    causal = {
        "associations": {
            "raw_score_validation": {
                "S_semantic_vs_G_semantic": {"rho": 0.8, "n": 4},
                "S_structural_vs_G_structural": {"rho": 0.7, "n": 4},
                "S_semantic_vs_G_structural_control": {"rho": 0.1, "n": 4},
                "S_structural_vs_G_semantic_control": {"rho": -0.1, "n": 4},
            },
            "J_vs_clean_prediction_movement": {
                "pooled": {"rho": 0.6, "n": 4},
                "within_layer_permutation": {"p": 0.04},
            },
            "J_vs_gross_total": {
                "pooled": {"rho": 0.5, "n": 4},
                "within_layer_permutation": {"p": 0.05},
            },
            "J_vs_necessity_total": {
                "pooled": {"rho": 0.4, "n": 4},
                "within_layer_permutation": {"p": 0.06},
            },
            "D_rel_vs_gross_contrast": {
                "pooled_active": {"rho": 0.9, "n": 3},
                "within_layer_permutation_active": {"p": 0.01},
            },
            "D_rel_vs_necessity_contrast": {
                "pooled_active": {"rho": 0.75, "n": 3},
                "within_layer_permutation_active": {"p": 0.02},
            },
        },
        "summary": {
            "targets": {
                "family_semantic_leaning": {
                    "calibrated": {
                        "g_semantic": 2.0,
                        "g_structural": 0.5,
                        "n_semantic": 1.5,
                        "n_structural": 0.5,
                    }
                },
                "family_structural_leaning": {
                    "calibrated": {
                        "g_semantic": 0.4,
                        "g_structural": 1.6,
                        "n_semantic": 0.3,
                        "n_structural": 1.3,
                    }
                },
                "control_semantic_leaning_central_control": {
                    "calibrated": {
                        "g_semantic": 0.8,
                        "g_structural": 0.7,
                        "n_semantic": 0.7,
                        "n_structural": 0.6,
                    }
                },
                "control_structural_leaning_central_control": {
                    "calibrated": {
                        "g_semantic": 0.7,
                        "g_structural": 0.8,
                        "n_semantic": 0.6,
                        "n_structural": 0.7,
                    }
                },
            }
        },
        "clean_ablation": {},
    }
    inputs = PaperInputs(
        score_bindings={},
        causal={("dense", 16, 0): causal},
        role_results={},
        counterfactual={},
        carriage={},
        best_seeds={},
        performance=(),
    )
    rows = causal_summary_rows(
        inputs,
        models=("dense",),
        causal_ns=(16,),
        seeds=(0,),
    )
    by_test = {
        row["test"]: row
        for row in rows
        if row["section"] == "coordinate_validation"
    }
    assert by_test["J_vs_clean_prediction_movement"]["estimate"] == pytest.approx(0.6)
    assert by_test["D_rel_vs_gross_contrast"]["estimate"] == pytest.approx(0.9)
    interaction = {
        row["test"]: row["estimate"]
        for row in rows
        if row["section"] == "family_interaction"
    }
    assert interaction["gross_family"] == pytest.approx(2.7)
    assert interaction["gross_control"] == pytest.approx(0.2)


def test_paper_frontend_defaults_are_cache_only_and_keep_N80_performance_only():
    args = build_parser().parse_args([])

    assert args.phase == "figures"
    assert args.score_ns == "4,8,16,32,64"
    assert args.causal_ns == "4,16,64"
    assert args.performance_ns == "4,8,16,32,64,80"
    assert args.paper_analysis_name != args.source_extension_name
