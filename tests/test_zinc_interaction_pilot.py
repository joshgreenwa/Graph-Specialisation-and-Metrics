import numpy as np
import pytest

import graph_specialisation_metrics.zinc_interaction_pilot as pilot_module
from graph_specialisation_metrics.zinc_interaction_pilot import (
    OUTPUT_MODULATION_TASKS,
    QM9_OUTPUT_MODULATION_TASKS,
    OutputModulationConfig,
    _write_csv,
    audit_output_modulation_pairing,
    carrier_alignment_analysis,
    carrier_alignment_specificity_analysis,
    event_distance_summary,
    figures_output_modulation,
    four_state_contrast,
    graph_absolute_distance_profiles,
    graph_distance_profiles,
    layer_event_summary,
    marginal_profile_diversity_analysis,
    output_modulation_graph_metrics,
    output_modulation_metrics,
    paired_reference_advantages,
    paired_reference_diversity_advantages,
)


def test_four_state_contrast_separates_additive_and_interacting_fields():
    clean = np.asarray([1.0, 2.0])
    semantic = np.asarray([2.0, 4.0])
    structural = np.asarray([3.0, 5.0])
    additive_joint = semantic + structural - clean
    np.testing.assert_allclose(
        four_state_contrast(clean, semantic, structural, additive_joint),
        np.zeros(2),
    )
    np.testing.assert_allclose(
        four_state_contrast(clean, semantic, structural, additive_joint + [0.0, 3.0]),
        [0.0, 3.0],
    )


@pytest.mark.parametrize(
    ("endpoints", "expected_m"),
    (
        ((10.0, 8.0, 7.0, 5.0), 0.0),
        ((10.0, 8.0, 7.0, 7.0), 2.0),
        ((10.0, 8.0, 7.0, 3.0), 2.0 / 3.0),
    ),
)
def test_output_modulation_m_has_exact_factorial_interpretation(endpoints, expected_m):
    result = output_modulation_metrics(*endpoints, effect_floor=1.0e-6)
    assert result["modulation_estimable"] is True
    assert result["modulation_m"] == pytest.approx(expected_m)
    assert 0.0 <= result["modulation_m"] <= 2.0


def test_output_modulation_m_does_not_normalise_a_null_semantic_effect():
    result = output_modulation_metrics(10.0, 10.0, 7.0, 7.0, effect_floor=1.0e-6)
    assert result["modulation_estimable"] is False
    assert np.isnan(result["modulation_m"])


def test_output_all_cli_accepts_legacy_and_output_task_defaults(tmp_path, monkeypatch):
    calls = []

    def fake_measure(config, **_kwargs):
        calls.append(("measure", tuple(config.tasks)))
        return {"events": 0}

    def fake_figures(config):
        calls.append(("figures", tuple(config.tasks)))
        return {"output_modulation_figures": {}}

    monkeypatch.setattr(pilot_module, "measure_output_modulation", fake_measure)
    monkeypatch.setattr(pilot_module, "figures_output_modulation", fake_figures)
    pilot_module.main(
        [
            "--phase",
            "output-all",
            "--output-dir",
            str(tmp_path),
            "--output-graphs",
            "1",
            "--output-sources-per-graph",
            "1",
            "--output-donor-pairs-per-source",
            "1",
            "--output-semantic-donor-graphs",
            "1",
            "--bootstrap-replicates",
            "2",
            "--skip-dependency-install",
        ]
    )
    assert calls == [
        ("measure", OUTPUT_MODULATION_TASKS),
        ("figures", OUTPUT_MODULATION_TASKS),
    ]


def test_output_validation_defaults_prioritise_graphs_then_sources():
    args = pilot_module.build_parser().parse_args(["--phase", "output-all"])
    assert args.output_graphs == 128
    assert args.output_sources_per_graph == 6
    assert args.output_donor_pairs_per_source == 2
    assert args.output_graphs_per_batch == 8


def test_qm9_output_modulation_suite_is_registered_separately(tmp_path):
    zinc = OutputModulationConfig(output_dir=tmp_path)
    qm9 = OutputModulationConfig(
        output_dir=tmp_path,
        tasks=QM9_OUTPUT_MODULATION_TASKS,
    )
    zinc.validate()
    qm9.validate()
    assert zinc.suite == "zinc"
    assert qm9.suite == "qm9"
    assert zinc.cache_dir != qm9.cache_dir


def test_output_all_cli_accepts_qm9_suite(tmp_path, monkeypatch):
    calls = []

    def fake_measure(config, **_kwargs):
        calls.append(("measure", config.suite, tuple(config.tasks)))
        return {"events": 0}

    def fake_figures(config):
        calls.append(("figures", config.suite, tuple(config.tasks)))
        return {"output_modulation_figures": {}}

    monkeypatch.setattr(pilot_module, "measure_output_modulation", fake_measure)
    monkeypatch.setattr(pilot_module, "figures_output_modulation", fake_figures)
    pilot_module.main(
        [
            "--phase",
            "output-all",
            "--output-dir",
            str(tmp_path),
            "--output-tasks",
            ",".join(QM9_OUTPUT_MODULATION_TASKS),
            "--output-graphs",
            "1",
            "--output-sources-per-graph",
            "1",
            "--output-donor-pairs-per-source",
            "1",
            "--output-semantic-donor-graphs",
            "1",
            "--bootstrap-replicates",
            "2",
            "--skip-dependency-install",
        ]
    )
    assert calls == [
        ("measure", "qm9", QM9_OUTPUT_MODULATION_TASKS),
        ("figures", "qm9", QM9_OUTPUT_MODULATION_TASKS),
    ]
    assert not (tmp_path / "analysis_config.json").exists()


def test_output_modulation_aggregates_pairs_then_sources_then_graphs():
    rows = []
    for source, values in ((0, (0.0, 2.0)), (1, (2.0, 2.0))):
        for pair, value in enumerate(values):
            rows.append(
                {
                    "task": "zinc_1hop",
                    "graph": 3,
                    "source": source,
                    "pair": pair,
                    "modulation_m": value,
                    "modulation_estimable": True,
                    "interaction_abs": value,
                    "semantic_reference_abs": 1.0,
                    "semantic_effect_original": 1.0,
                    "semantic_effect_swapped_structure": 1.0,
                }
            )
    metrics = output_modulation_graph_metrics(rows)
    modulation = next(row for row in metrics if row["metric"] == "modulation_m")
    assert modulation["value"] == pytest.approx(1.5)
    assert modulation["eligible_sources"] == 2


def test_output_modulation_pairing_audit_requires_identical_donors():
    tasks = OUTPUT_MODULATION_TASKS
    rows = [
        {
            "task": task,
            "graph": 3,
            "source": 1,
            "pair": 0,
            "semantic_donor_graph": 9,
            "semantic_donor_node": 2,
            "structural_donor_node": 4,
        }
        for task in tasks
    ]
    assert audit_output_modulation_pairing(rows, tasks=tasks) == {
        "paired_events": 1,
        "models": 6,
    }
    rows[-1]["structural_donor_node"] = 5
    with pytest.raises(RuntimeError, match="donor identities differ"):
        audit_output_modulation_pairing(rows, tasks=tasks)


def test_output_modulation_figures_rebuild_from_cached_endpoints(tmp_path):
    config = OutputModulationConfig(
        output_dir=tmp_path,
        graphs=2,
        sources_per_graph=1,
        donor_pairs_per_source=1,
        semantic_donor_graphs=2,
        bootstrap_replicates=10,
    )
    joint_by_task = {
        "zinc_1hop_localrrwp": 6.0,
        "zinc_1hop": 5.0,
        "zinc_1hop_vnode": 5.2,
        "zinc_2hop": 5.5,
        "zinc_2hop_vnode": 5.4,
        "zinc": 4.0,
    }
    rows = []
    for task in OUTPUT_MODULATION_TASKS:
        for graph in range(2):
            endpoints = (10.0, 8.0, 7.0, joint_by_task[task])
            rows.append(
                {
                    "analysis_version": "test",
                    "fingerprint": config.fingerprint,
                    "task": task,
                    "graph": graph,
                    "source": 0,
                    "pair": 0,
                    "semantic_donor_graph": 9,
                    "semantic_donor_node": 2,
                    "semantic_dose": 1.0,
                    "structural_donor_node": 4,
                    "structural_dose": 1.0,
                    "output_clean": endpoints[0],
                    "output_semantic": endpoints[1],
                    "output_structural": endpoints[2],
                    "output_joint": endpoints[3],
                    **output_modulation_metrics(*endpoints, effect_floor=config.effect_floor),
                }
            )
    _write_csv(config.cache_dir / "output_modulation_events.csv", rows)
    result = figures_output_modulation(config)
    figure = result["output_modulation_figures"]["output_modulation"]
    assert result["output_modulation_pairing"]["paired_events"] == 2
    assert (config.cache_dir / "output_modulation_summary.csv").is_file()
    assert figure["png"].endswith(".png")

    high_floor = OutputModulationConfig(
        output_dir=tmp_path,
        graphs=2,
        sources_per_graph=1,
        donor_pairs_per_source=1,
        semantic_donor_graphs=2,
        effect_floor=100.0,
        bootstrap_replicates=10,
    )
    assert high_floor.cache_dir == config.cache_dir
    rebuilt = figures_output_modulation(high_floor)
    assert not any(row["metric"] == "modulation_m" for row in rebuilt["output_modulation_summary"])


def test_output_modulation_imports_compatible_smaller_task_cache(tmp_path):
    config = OutputModulationConfig(output_dir=tmp_path)
    legacy_dir = tmp_path / "output_modulation" / "legacy-cache"
    pilot_module._write_json(
        legacy_dir / "output_modulation_config.json",
        {
            "seed": config.seed,
            "sources_per_graph": config.sources_per_graph,
            "donor_pairs_per_source": config.donor_pairs_per_source,
            "semantic_donor_graphs": config.semantic_donor_graphs,
            "analysis_seed": config.analysis_seed,
            "graphs": 64,
            "tasks": ["zinc_1hop_localrrwp", "zinc_1hop", "zinc"],
        },
    )
    legacy_event = {
        "analysis_version": pilot_module.OUTPUT_MODULATION_VERSION,
        "fingerprint": "legacy",
        "task": "zinc_1hop_localrrwp",
        "graph": 3,
        "source": 1,
        "pair": 0,
        "semantic_donor_graph": 9,
        "semantic_donor_node": 2,
        "structural_donor_node": 4,
        "output_clean": 10.0,
        "output_semantic": 8.0,
        "output_structural": 7.0,
        "output_joint": 6.0,
    }
    _write_csv(legacy_dir / "output_modulation_events.csv", [legacy_event])
    _write_csv(
        legacy_dir / "completed_graphs.csv",
        [{"fingerprint": "legacy", "task": "zinc_1hop_localrrwp", "graph": 3}],
    )
    events, completed, _health, imported = pilot_module._merge_compatible_output_caches(
        config,
        events=[],
        completed=[],
        health=[],
    )
    assert imported == [str(legacy_dir)]
    assert events[0]["fingerprint"] == config.fingerprint
    assert completed == [
        {
            "fingerprint": config.fingerprint,
            "task": "zinc_1hop_localrrwp",
            "graph": 3,
        }
    ]


def test_distance_summary_does_not_normalise_a_null_interaction():
    null = event_distance_summary(
        np.asarray([1.0, 2.0]),
        np.asarray([1.5, 1.5]),
        np.asarray([1.0e-7, 2.0e-7]),
        np.asarray([1.0, 5.0]),
        absolute_floor=1.0e-6,
        relative_floor=1.0e-3,
        far_distance=4,
    )
    assert null["interaction_estimable"] is False
    assert np.isnan(null["interaction_expected_distance"])
    effect = event_distance_summary(
        np.asarray([1.0, 2.0]),
        np.asarray([1.5, 1.5]),
        np.asarray([0.0, 1.0]),
        np.asarray([1.0, 5.0]),
        absolute_floor=1.0e-6,
        relative_floor=1.0e-3,
        far_distance=4,
    )
    assert effect["interaction_estimable"] is True
    assert effect["interaction_expected_distance"] == pytest.approx(5.0)
    assert effect["interaction_far_share"] == pytest.approx(1.0)


def test_layer_summary_keeps_virtual_carriage_out_of_physical_distance():
    summary = layer_event_summary(
        np.asarray([1.0, 1.0, 2.0]),
        np.asarray([2.0, 0.0, 2.0]),
        np.asarray([0.0, 1.0, 3.0]),
        np.asarray([0.0, 4.0]),
        absolute_floor=0.1,
        relative_floor=0.01,
        far_distance=4,
        virtual_index=2,
    )
    assert summary["interaction_estimable"] is True
    assert summary["interaction_distance_estimable"] is True
    assert summary["semantic_expected_distance"] == pytest.approx(2.0)
    assert summary["interaction_expected_distance"] == pytest.approx(4.0)
    assert summary["interaction_far_share"] == pytest.approx(0.25)
    assert summary["interaction_virtual_share"] == pytest.approx(0.75)


def test_graph_profiles_average_pairs_then_sources_and_skip_null_interactions():
    rows = []
    for source, pair, masses, eligible in (
        (0, 0, (1.0, 0.0), True),
        (0, 1, (0.0, 1.0), False),
        (1, 0, (1.0, 0.0), True),
    ):
        for distance, semantic_mass in enumerate(masses):
            rows.append(
                {
                    "task": "zinc_1hop",
                    "graph": 3,
                    "source": source,
                    "pair": pair,
                    "distance": distance,
                    "semantic_mass": semantic_mass,
                    "structural_mass": semantic_mass,
                    "interaction_mass": semantic_mass,
                    "interaction_estimable": eligible,
                }
            )
    profiles = graph_distance_profiles(rows, layerwise=False)
    semantic = {int(row["distance"]): row["share"] for row in profiles if row["term"] == "semantic"}
    interaction = {
        int(row["distance"]): row["share"] for row in profiles if row["term"] == "interaction"
    }
    assert semantic == pytest.approx({0: 0.75, 1: 0.25})
    assert interaction == pytest.approx({0: 1.0, 1: 0.0})


def test_absolute_profiles_retain_scale_instead_of_allocation_share():
    rows = [
        {
            "task": "zinc_1hop",
            "graph": 3,
            "source": 0,
            "pair": 0,
            "carrier": distance,
            "distance": distance,
            "semantic_mass": mass,
            "structural_mass": 2.0 * mass,
            "interaction_mass": 4.0 * mass,
            "interaction_estimable": True,
        }
        for distance, mass in enumerate((1.0, 3.0))
    ]
    profiles = graph_absolute_distance_profiles(rows)
    interaction = {
        int(row["distance"]): row["mass"] for row in profiles if row["term"] == "interaction"
    }
    assert interaction == pytest.approx({0: 4.0, 1: 12.0})


def test_carrier_alignment_distinguishes_signed_fit_from_mass_only_screen():
    rows = []
    for carrier, values in enumerate(
        (
            (1.0, 1.0, 2.0, -1.0, 1.0, 2.0),
            (1.0, 2.0, 4.0, 1.0, -2.0, -4.0),
        )
    ):
        sem_mass, struct_mass, int_mass, sem_q, struct_q, int_q = values
        rows.append(
            {
                "task": "zinc_1hop",
                "graph": 3,
                "source": 0,
                "pair": 0,
                "carrier": carrier,
                "distance": carrier,
                "semantic_mass": sem_mass,
                "structural_mass": struct_mass,
                "interaction_mass": int_mass,
                "semantic_projection": sem_q,
                "structural_projection": struct_q,
                "interaction_projection": int_q,
                "interaction_estimable": True,
            }
        )
    metrics, residuals, modes = carrier_alignment_analysis(rows)
    assert modes == ("absolute_mass", "signed_projection")
    structural = {
        row["metric"]: row["value"]
        for row in metrics
        if row["mode"] == "signed_projection" and row["reference"] == "structural"
    }
    assert structural["cosine"] == pytest.approx(1.0)
    assert structural["gain"] == pytest.approx(2.0)
    assert structural["explained_energy"] == pytest.approx(1.0)
    assert structural["residual_fraction"] == pytest.approx(0.0)
    assert all(
        row["residual_mass"] == pytest.approx(0.0)
        for row in residuals
        if row["mode"] == "signed_projection" and row["reference"] == "structural"
    )

    legacy_rows = [
        {key: value for key, value in row.items() if not key.endswith("_projection")}
        for row in rows
    ]
    _, _, legacy_modes = carrier_alignment_analysis(legacy_rows)
    assert legacy_modes == ("absolute_mass",)


def test_same_source_shuffle_separates_event_specific_fit_from_locality_envelope():
    rows = []
    structural_profiles = ((1.0, 0.0), (0.0, 1.0), (1.0, 1.0))
    semantic_profiles = ((1.0, 1.0),) * 3
    for pair in range(3):
        for carrier in range(2):
            rows.append(
                {
                    "task": "zinc_1hop",
                    "graph": 3,
                    "source": 0,
                    "pair": pair,
                    "carrier": carrier,
                    "distance": carrier,
                    "semantic_mass": semantic_profiles[pair][carrier],
                    "structural_mass": structural_profiles[pair][carrier],
                    "interaction_mass": structural_profiles[pair][carrier],
                    "interaction_estimable": True,
                }
            )
    alignment, _residuals, _modes = carrier_alignment_analysis(rows)
    specificity = carrier_alignment_specificity_analysis(rows)
    structural = next(
        row
        for row in specificity
        if row["reference"] == "structural"
        and row["metric"] == "explained_energy"
        and row["analysis"] == "matched_minus_shuffle"
    )
    semantic = next(
        row
        for row in specificity
        if row["reference"] == "semantic"
        and row["metric"] == "explained_energy"
        and row["analysis"] == "matched_minus_shuffle"
    )
    assert structural["value"] == pytest.approx(2.0 / 3.0)
    assert semantic["value"] == pytest.approx(0.0)

    structural_normalized = next(
        row
        for row in specificity
        if row["reference"] == "structural"
        and row["metric"] == "explained_energy"
        and row["analysis"] == "headroom_normalized_advantage"
    )
    semantic_normalized = next(
        row
        for row in specificity
        if row["reference"] == "semantic"
        and row["metric"] == "explained_energy"
        and row["analysis"] == "headroom_normalized_advantage"
    )
    assert structural_normalized["value"] == pytest.approx(1.0)
    assert semantic_normalized["value"] == pytest.approx(0.0)

    diversity = marginal_profile_diversity_analysis(rows)
    structural_diversity = next(row for row in diversity if row["reference"] == "structural")
    semantic_diversity = next(row for row in diversity if row["reference"] == "semantic")
    assert structural_diversity["value"] == pytest.approx(2.0 / 3.0)
    assert semantic_diversity["value"] == pytest.approx(0.0)

    paired = [
        *paired_reference_advantages(alignment, specificity),
        *paired_reference_diversity_advantages(diversity),
    ]
    actual = next(
        row
        for row in paired
        if row["metric"] == "explained_energy" and row["comparison"] == "actual_fit"
    )
    event_specificity = next(
        row
        for row in paired
        if row["metric"] == "explained_energy" and row["comparison"] == "event_specificity"
    )
    assert actual["value"] == pytest.approx(1.0 / 3.0)
    assert event_specificity["value"] == pytest.approx(2.0 / 3.0)
    normalized_specificity = next(
        row
        for row in paired
        if row["metric"] == "explained_energy"
        and row["comparison"] == "normalized_event_specificity"
    )
    diversity_advantage = next(
        row
        for row in paired
        if row["metric"] == "explained_energy" and row["comparison"] == "reference_diversity"
    )
    assert normalized_specificity["value"] == pytest.approx(1.0)
    assert diversity_advantage["value"] == pytest.approx(2.0 / 3.0)
