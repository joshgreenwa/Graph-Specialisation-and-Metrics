from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from graph_specialisation_metrics.specialisation.paper_figures import (
    _ci_status,
    _figure_transport,
    _reconstruction_state,
    _score_coordinates,
    _spearman_rho,
    _transport_arrays,
    _transport_rows,
    make_paper_figures,
)


def _result_fixture():
    return {
        "title": "ZINC fixture",
        "S_sem": np.array([[2.0, 0.5], [1.2, 0.8]]),
        "S_str": np.array([[0.5, 2.0], [0.9, 1.1]]),
        "L": 2,
        "H": 2,
    }


def _mediation_fixture():
    rows = []
    cell_values = {
        "noising": {
            ("semantic", "semantic"): 0.72,
            ("semantic", "rrwp_role"): 0.18,
            ("rrwp_role", "semantic"): 0.12,
            ("rrwp_role", "rrwp_role"): 0.61,
        },
        "denoising": {
            ("semantic", "semantic"): 0.66,
            ("semantic", "rrwp_role"): 0.15,
            ("rrwp_role", "semantic"): 0.17,
            ("rrwp_role", "rrwp_role"): 0.58,
        },
    }
    for direction, values in cell_values.items():
        for (selector, channel), beta in values.items():
            rows.append(
                {
                    "group_id": f"{selector}-4",
                    "selector": selector,
                    "k": 4,
                    "channel": channel,
                    "direction": direction,
                    "beta": beta,
                    "loss_effect_mean": beta * 0.01,
                    "n_graphs": 32,
                }
            )
        # Multiple matched draws exercise the plotter's deterministic averaging path.
        for draw in range(2):
            for channel in ("semantic", "rrwp_role"):
                rows.append(
                    {
                        "group_id": f"control-{draw}",
                        # Production keeps the channel selector and distinguishes controls here.
                        "selector": "semantic",
                        "group_type": "matched_control",
                        "control_draw": draw,
                        "k": 4,
                        "channel": channel,
                        "direction": direction,
                        "beta": 0.03 + 0.01 * draw,
                    }
                )

    topk = []
    for direction in ("noising", "denoising"):
        for k, estimate in ((1, 0.18), (2, 0.31), (4, 0.52)):
            topk.append(
                {
                    "k": k,
                    "direction": direction,
                    "double_dissociation": estimate,
                    "ci_low": estimate - 0.08,
                    "ci_high": estimate + 0.08,
                    "matched_control_mean": 0.01,
                    "matched_control_ci_low": -0.05,
                    "matched_control_ci_high": 0.05,
                }
            )
    return {
        "calibration": {
            "importance": np.array([[2.1, 2.0], [1.1, 1.0]]),
            "preference": np.array([[0.7, -0.8], [0.2, -0.1]]),
            "eligible": np.ones((2, 2), dtype=bool),
            "rankings": {
                "semantic": [[0, 0], [1, 0]],
                "rrwp_role": [[0, 1], [1, 1]],
            },
        },
        "groups": {"summary": rows},
        "contrasts": {
            "topk": topk,
            "primary": [row for row in topk if row["k"] == 4],
            "interactions": [
                {
                    "selector": "semantic",
                    "k": 4,
                    "channel": "semantic",
                    "direction": "noising",
                    "group_beta": 0.72,
                    "summed_single_beta": 0.64,
                    "interaction_beta": 0.08,
                    "ci_low": 0.01,
                    "ci_high": 0.15,
                }
            ],
        },
        "bootstrap_replicates": 200,
    }


def _summary(mean, lo, hi):
    return {"mean": mean, "ci_low": lo, "ci_high": hi, "n_graphs": 32}


def _transport_fixture(valid=True):
    causal = {}
    for group, scale in (("semantic", 1.0), ("rrwp_role", 0.85), ("matched_control", 0.15)):
        causal[group] = {}
        for k in (1, 2, 4):
            causal[group][str(k)] = {
                "static_routing": {
                    "delta_mae": _summary(scale * 0.002 * k, scale * 0.001 * k, scale * 0.003 * k)
                },
                "broadcast_only": {
                    "delta_mae": _summary(scale * 0.004 * k, scale * 0.002 * k, scale * 0.006 * k)
                },
                "residual_permuted": {
                    "delta_mae": _summary(scale * 0.003 * k, scale * 0.001 * k, scale * 0.005 * k)
                },
                "full_head_zero": {
                    "delta_mae": _summary(scale * 0.006 * k, scale * 0.003 * k, scale * 0.009 * k)
                },
                "remove_content_residual": {
                    "delta_mae": _summary(scale * 0.001 * k, -0.001, scale * 0.003 * k)
                },
                "remove_edge_residual": {
                    "delta_mae": _summary(scale * 0.002 * k, 0.0002, scale * 0.004 * k)
                },
            }
    return {
        "primary_k": 4,
        "topk": np.array([1, 2, 4]),
        "head_selection": {
            "semantic": [[0, 0], [1, 0]],
            "rrwp_role": [[0, 1], [1, 1]],
        },
        "head_metrics": {
            "routing_js_static": np.array([[0.8, 0.7], [0.5, 0.4]]),
            "routing_l2_relative": np.array([[0.2, 0.3], [0.5, 0.6]]),
            "effective_relational_fraction": np.array([[0.7, 0.6], [0.4, 0.5]]),
        },
        "causal": causal,
        "checks": {
            "passed": valid,
            "max_wv_reconstruction_error": 2.0e-7 if valid else 2.0e-2,
        },
    }


def test_make_paper_figures_writes_exactly_two_stems_per_task(tmp_path):
    figures_before = set(plt.get_fignums())
    output = make_paper_figures(
        {"zinc": _result_fixture()},
        {"zinc": _mediation_fixture()},
        {"zinc": _transport_fixture()},
        tmp_path,
    )

    assert set(output) == {"zinc"}
    assert set(output["zinc"]) == {
        "channel_causal_mediation",
        "effective_relational_transport",
    }
    expected = {
        "fig_channel_causal_mediation_zinc.png",
        "fig_channel_causal_mediation_zinc.pdf",
        "fig_effective_relational_transport_zinc.png",
        "fig_effective_relational_transport_zinc.pdf",
    }
    produced = {path.name for path in (tmp_path / "paper").iterdir()}
    assert produced == expected
    for formats in output["zinc"].values():
        assert set(formats) == {"png", "pdf"}
        for path in formats.values():
            artifact = Path(path)
            assert artifact.exists()
            assert artifact.stat().st_size > 1_000
    assert set(plt.get_fignums()) == figures_before


def test_nested_extension_schema_is_read_without_copying_model_tensors():
    result = _result_fixture()
    mediation = _mediation_fixture()
    importance, preference = _score_coordinates(result, mediation)
    assert np.allclose(importance, [2.1, 2.0, 1.1, 1.0])
    assert np.allclose(preference, [0.7, -0.8, 0.2, -0.1])

    transport = _transport_fixture()
    routing, relational, label = _transport_arrays(transport)
    assert label == "receiver-specific routing\n(JS from matched static null)"
    assert np.allclose(routing, [0.8, 0.7, 0.5, 0.4])
    assert np.allclose(relational, [0.7, 0.6, 0.4, 0.5])
    assert _spearman_rho(routing, relational) == pytest.approx(0.8)

    rows = _transport_rows(transport)
    assert len(rows) == 3 * 3 * 6
    assert {row["_group"] for row in rows} == {"semantic", "rrwp", "control"}
    assert {row["_intervention"] for row in rows} >= {
        "staticise_attention",
        "broadcast_only",
        "residual_permuted",
        "head_zero",
        "remove_content_residual",
        "remove_edge_residual",
    }


def test_null_and_invalid_reconstruction_are_explicit(tmp_path):
    mediation = _mediation_fixture()
    for row in mediation["contrasts"]["topk"]:
        row["ci_low"] = -0.1
        row["ci_high"] = 0.1
    assert _ci_status(0.01, -0.1, 0.1, noun="double dissociation") == (
        "No resolved double dissociation"
    )

    transport = _transport_fixture(valid=False)
    assert _reconstruction_state(transport) == (False, 2.0e-2)
    figure = _figure_transport("zinc", _result_fixture(), mediation, transport)
    assert all(
        any("Analysis blocked" in text.get_text() for text in axis.texts)
        for axis in figure.axes
    )
    plt.close(figure)
    output = make_paper_figures(
        {"zinc": _result_fixture()},
        {"zinc": mediation},
        {"zinc": transport},
        tmp_path,
    )
    assert Path(output["zinc"]["effective_relational_transport"]["png"]).exists()


def test_minimal_missing_extension_results_render_failure_panels(tmp_path):
    output = make_paper_figures(
        {"zinc": {"title": "Minimal ZINC", "S_sem": [[1.0]], "S_str": [[1.0]]}},
        {},
        {},
        tmp_path,
    )
    assert len(output["zinc"]) == 2
    assert len(list((tmp_path / "paper").glob("*"))) == 4
