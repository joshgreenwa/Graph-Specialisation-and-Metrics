from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np

from graph_specialisation_metrics.methodology.bootstrap import (
    Observation,
    nested_percentile_interval,
)
from graph_specialisation_metrics.methodology.graphormer_causal_analysis import (
    production_config,
)
from graph_specialisation_metrics.methodology.graphormer_causal_population import (
    PopulationPolicy,
    _causal_preference_analysis,
    _event_interval,
    build_population_gate,
    layer_adjusted_components,
    response_adjusted_components,
)
from graph_specialisation_metrics.methodology.graphormer_causal_population_plots import (
    render_population_figure_suite,
)
from graph_specialisation_metrics.methodology.scores import HeadCoordinates


def test_nested_bootstrap_progress_callback_reports_all_draws():
    calls = []
    observations = [
        Observation(seed=0, graph=graph, source=0, donor=0, value=float(graph))
        for graph in range(10)
    ]
    config = production_config(
        output_dir="/tmp/bootstrap-progress",
        dataset_root="/tmp/pcqm",
        cache_dir="/tmp/hf",
        accelerator="cpu",
    )
    interval = nested_percentile_interval(
        observations,
        config.bootstrap,
        on_draw=lambda completed, total: calls.append((completed, total)),
    )
    assert interval.replicates == 2_000
    assert calls[0] == (1, 2_000)
    assert calls[-1] == (2_000, 2_000)
    assert len(calls) == 2_000


def _scores():
    J = np.asarray(
        [
            [1.10, 1.00, 0.92, 0.86, 1.08, 0.98, 0.90, 0.84, 0.95, 0.88],
            [1.20, 1.06, 0.96, 0.82, 1.18, 1.04, 0.94, 0.80, 1.00, 0.90],
            [1.14, 1.02, 0.89, 0.78, 1.12, 1.00, 0.87, 0.76, 0.93, 0.85],
        ]
    )
    D = np.asarray(
        [
            [0.48, 0.42, 0.36, 0.30, -0.47, -0.41, -0.35, -0.29, 0.02, -0.03],
            [0.51, 0.43, 0.34, 0.28, -0.49, -0.40, -0.33, -0.27, 0.04, -0.01],
            [0.46, 0.39, 0.32, 0.26, -0.45, -0.38, -0.31, -0.25, 0.01, -0.05],
        ]
    )
    semantic = J * (1 + D)
    structural = J * (1 - D)
    coordinates = HeadCoordinates(
        raw_semantic=semantic,
        raw_structural=structural,
        semantic_mean=1.0,
        structural_mean=1.0,
        normalized_semantic=semantic,
        normalized_structural=structural,
        joint_sensitivity=J,
        selectivity=D,
        active=np.ones_like(J, dtype=bool),
        estimable=True,
    )
    return {"coordinates": coordinates}


def _confidence_gate():
    return {
        "preference_threshold": 0.10,
        "activity_floor": 0.20,
        "heads": {
            "semantic_specialist": ((0, 0), (1, 0), (2, 0)),
            "structural_specialist": ((0, 4), (1, 4), (2, 4)),
        },
    }


def test_production_config_accepts_paper_scale_disjoint_populations(tmp_path):
    config = production_config(
        output_dir=str(tmp_path),
        dataset_root=str(tmp_path / "pcqm"),
        cache_dir=str(tmp_path / "hf"),
        accelerator="cpu",
        discovery_graphs=256,
        causal_graphs=256,
        clean_ablation_graphs=256,
        semantic_donor_graphs=2_000,
        sources_per_graph=6,
        donors_per_source=8,
        graphs_per_batch=16,
    )
    config.validate()
    assert config.sizes.discovery_graphs == 256
    assert config.sizes.causal_graphs == 256
    assert config.sizes.clean_ablation_graphs == 256
    assert config.sizes.semantic_donor_graphs == 2_000
    assert config.sizes.sources_per_graph == 6
    assert config.sizes.donors_per_source == 8
    assert config.execution.graphs_per_batch == 16


def test_population_gate_is_discovery_only_balanced_and_nonoverlapping():
    gate = build_population_gate(
        _scores(),
        _confidence_gate(),
        PopulationPolicy(head_pairs=3, minimum_pairs=2, candidate_pool_multiplier=2),
    )
    assert gate["status"] == "estimable"
    assert len(gate["specialist_pairs"]) == 3
    assert len(gate["null_pairs"]) == 6
    semantic = set(gate["heads"]["semantic"])
    structural = set(gate["heads"]["structural"])
    null = set(gate["heads"]["j_matched_null"])
    assert semantic.isdisjoint(structural)
    assert semantic.isdisjoint(null)
    assert structural.isdisjoint(null)
    assert not gate["selection_uses_causal_outcomes"]
    assert (
        abs(gate["matching_balance"]["semantic_vs_structural"]["standardized_J_difference"]) < 0.5
    )


def test_layer_adjusted_partial_slope_equals_fixed_effect_regression():
    layers = np.repeat(np.arange(3), 8)
    x = np.linspace(-1.5, 2.0, len(layers)) + 0.15 * layers
    y = 0.8 * x + np.asarray([0.0, 1.5, -1.0])[layers]
    result = layer_adjusted_components(x, y, layers)

    x_scaled = (x - np.mean(x)) / np.std(x)
    y_scaled = (y - np.mean(y)) / np.std(y)
    indicators = np.column_stack(((layers == 1).astype(float), (layers == 2).astype(float)))
    design = np.column_stack((np.ones(len(x)), x_scaled, indicators))
    expected = float(np.linalg.lstsq(design, y_scaled, rcond=None)[0][1])
    assert np.isclose(result["beta"], expected)
    assert set(result["leave_one_layer_out"]) == {"0", "1", "2"}


def test_response_adjusted_slope_matches_direct_regression():
    layers = np.repeat(np.arange(3), 10)
    selectivity = np.linspace(-0.5, 0.6, len(layers)) + 0.03 * layers
    response = 0.7 + 0.2 * np.sin(np.arange(len(layers))) + 0.02 * layers
    preference = (
        0.7 * selectivity
        + 0.4 * response
        + np.asarray([0.0, 0.8, -0.5])[layers]
    )
    result = response_adjusted_components(
        selectivity, preference, response, layers
    )

    scaled = [
        (values - np.mean(values)) / np.std(values)
        for values in (selectivity, preference, response)
    ]
    x_scaled, y_scaled, response_scaled = scaled
    indicators = np.column_stack(
        ((layers == 1).astype(float), (layers == 2).astype(float))
    )
    design = np.column_stack(
        (np.ones(len(selectivity)), x_scaled, response_scaled, indicators)
    )
    expected = float(np.linalg.lstsq(design, y_scaled, rcond=None)[0][1])
    assert np.isclose(result["beta"], expected)


def test_population_interval_resamples_complete_matched_head_families(tmp_path):
    config = production_config(
        output_dir=str(tmp_path),
        dataset_root=str(tmp_path / "pcqm"),
        cache_dir=str(tmp_path / "hf"),
        accelerator="cpu",
    )
    gate = {
        "specialist_pairs": (
            {"semantic": (0, 0), "structural": (0, 1)},
            {"semantic": (1, 0), "structural": (1, 1)},
        ),
        "null_pairs": (
            {"target": (0, 0), "null": (0, 2)},
            {"target": (0, 1), "null": (0, 3)},
            {"target": (1, 0), "null": (1, 2)},
            {"target": (1, 1), "null": (1, 3)},
        ),
        "heads": {
            "semantic": ((0, 0), (1, 0)),
            "structural": ((0, 1), (1, 1)),
            "j_matched_null": ((0, 2), (0, 3), (1, 2), (1, 3)),
        },
    }
    rows = []
    all_heads = tuple(
        gate["heads"]["semantic"] + gate["heads"]["structural"] + gate["heads"]["j_matched_null"]
    )
    for channel_position, channel in enumerate(("semantic", "structural")):
        for graph in range(3):
            for source in range(2):
                for donor in range(2):
                    for head_position, head in enumerate(all_heads):
                        rows.append(
                            {
                                "channel": channel,
                                "graph": graph,
                                "source": source,
                                "donor": donor,
                                "head": head,
                                "controlled": True,
                                "N_fraction": (
                                    0.01 * head_position + 0.05 * channel_position + 0.002 * graph
                                ),
                            }
                        )
    summary = _event_interval(
        rows,
        gate,
        config,
        metric_order=("N_fraction",),
        require_controlled=False,
        seed_offset=7,
    )
    assert summary["estimate"].shape == (3, 2, 1)
    assert summary["low"].shape == summary["estimate"].shape
    assert summary["correct_pairing_advantage"].shape == (1,)
    assert summary["correct_pairing_low"].shape == (1,)
    assert summary["correct_pairing_high"].shape == (1,)
    assert summary["head_estimate"].shape == (2, 8, 1)
    assert summary["head_draws"].shape == (2_000, 2, 8, 1)
    assert summary["head_pair_count"] == 2
    assert summary["null_head_count"] == 4
    assert "matched head pair" in summary["resampled_levels"]


def test_causal_preference_uses_semantic_minus_structural_effect(tmp_path):
    config = production_config(
        output_dir=str(tmp_path),
        dataset_root=str(tmp_path / "pcqm"),
        cache_dir=str(tmp_path / "hf"),
        accelerator="cpu",
    )
    gate = {
        "specialist_pairs": (
            {"semantic": (0, 0), "structural": (0, 4)},
            {"semantic": (1, 0), "structural": (1, 4)},
        ),
        "null_pairs": (
            {"target": (0, 0), "null": (0, 8)},
            {"target": (0, 4), "null": (0, 9)},
            {"target": (1, 0), "null": (1, 8)},
            {"target": (1, 4), "null": (1, 9)},
        ),
    }
    head_order = tuple(
        sorted(
            {
                tuple(row[key])
                for row in gate["specialist_pairs"]
                for key in ("semantic", "structural")
            }
            | {tuple(row["null"]) for row in gate["null_pairs"]}
        )
    )
    scores = _scores()
    coordinates = scores["coordinates"]
    selectivity = np.asarray(
        [coordinates.selectivity[head] for head in head_order]
    )
    response = np.asarray(
        [0.02 + 0.003 * coordinates.joint_sensitivity[head] for head in head_order]
    )
    point = np.empty((2, len(head_order), 2), dtype=np.float64)
    point[0, :, 0] = response + 0.006 * selectivity
    point[1, :, 0] = response - 0.006 * selectivity
    point[0, :, 1] = response + 0.004 * selectivity
    point[1, :, 1] = response - 0.004 * selectivity
    rng = np.random.default_rng(5)
    draws = point[None] + rng.normal(0.0, 1.0e-4, size=(40, *point.shape))
    result = _causal_preference_analysis(
        {
            "head_order": head_order,
            "head_estimate": point,
            "head_draws": draws,
            "metric_order": ("R_align_matched", "I_align_matched"),
            "resampled_levels": ("graph", "source", "donor", "matched head pair"),
        },
        scores,
        gate,
        config,
    )
    expected_restoration = 0.012 * selectivity
    assert np.allclose(
        result["endpoints"]["restoration"]["causal_preference"],
        expected_restoration,
    )
    assert result["endpoints"]["restoration"]["spearman_rho"] > 0.99
    assert "matched head pair" not in result["resampled_levels"]
    assert "matched four-head block" in result["resampled_levels"]


def test_population_figures_export_vector_pdf_and_600_dpi_png(tmp_path):
    scores = _scores()
    gate = build_population_gate(
        scores,
        _confidence_gate(),
        PopulationPolicy(head_pairs=3, minimum_pairs=2, candidate_pool_multiplier=2),
    )
    estimate = np.asarray([[0.10, 0.04], [0.03, 0.11]])
    necessity = np.asarray([[0.20, 0.08], [0.07, 0.22], [0.05, 0.05]])
    J = scores["coordinates"].joint_sensitivity.reshape(-1)
    layers = np.repeat(np.arange(3), 10)
    movement = 0.02 + 0.03 * J + 0.003 * layers
    partial = layer_adjusted_components(J, movement, layers)

    def endpoint(values, spread):
        pairing = float((values[0, 0] - values[0, 1]) - (values[1, 0] - values[1, 1]))
        return {
            "estimate": values,
            "low": values - spread,
            "high": values + spread,
            "correct_pairing_advantage": pairing,
            "correct_pairing_low": pairing - spread,
            "correct_pairing_high": pairing + spread,
        }

    core = {
        "population_gate": gate,
        "raw_restoration": endpoint(estimate * 1.2, 0.02),
        "raw_injection": endpoint(estimate, 0.02),
        "restoration": endpoint(estimate * 0.8, 0.02),
        "injection": endpoint(estimate * 0.7, 0.02),
        "necessity": endpoint(necessity, 0.03),
        "clean_ablation": {
            "J": J,
            "layers": layers,
            "prediction_movement": movement,
            "prediction_movement_low": movement - 0.002,
            "prediction_movement_high": movement + 0.002,
            "spearman_rho": 0.53,
            "spearman_low": 0.48,
            "spearman_high": 0.55,
            "layer_adjusted_standardized_beta": partial["beta"],
            "layer_adjusted_low": partial["beta"] - 0.05,
            "layer_adjusted_high": partial["beta"] + 0.05,
            "head_count": len(J),
            "graph_count": 128,
            "layer_adjusted_partial": partial,
        },
        "causal_preference": {
            "selectivity": scores["coordinates"].selectivity.reshape(-1),
            "layers": layers,
            "head_count": len(J),
            "graph_count": 128,
            "endpoints": {
                name: {
                    "causal_preference": values,
                    "preference_low": values - 0.015,
                    "preference_high": values + 0.015,
                    "spearman_rho": 0.46 + 0.04 * position,
                    "spearman_low": 0.35 + 0.04 * position,
                    "spearman_high": 0.56 + 0.04 * position,
                    "response_layer_adjusted_beta": 0.62 + 0.05 * position,
                    "response_layer_adjusted_low": 0.48 + 0.05 * position,
                    "response_layer_adjusted_high": 0.75 + 0.05 * position,
                }
                for position, (name, values) in enumerate(
                    (
                        (
                            "restoration",
                            0.05
                            * scores["coordinates"].selectivity.reshape(-1)
                            + 0.002 * layers,
                        ),
                        (
                            "injection",
                            0.04
                            * scores["coordinates"].selectivity.reshape(-1)
                            + 0.001 * layers,
                        ),
                    )
                )
            },
        },
    }
    outputs = render_population_figure_suite(
        scores,
        gate,
        core,
        output_dir=tmp_path,
        common_metadata={"test": True},
    )
    assert set(outputs) == {
        "population_causal_tests_raw",
        "population_causal_tests_mismatch_adjusted",
        "correct_pairing_advantage",
        "J_vs_clean_ablation",
        "Drel_vs_causal_preference",
        "population_selection",
    }
    for paths in outputs.values():
        assert {Path(path).suffix for path in paths} == {".pdf", ".png"}
        assert all(Path(path).stat().st_size > 0 for path in paths)
        assert Path(paths[0]).with_suffix(".metadata.json").is_file()
