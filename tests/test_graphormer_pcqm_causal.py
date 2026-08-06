from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import numpy as np

from graph_specialisation_metrics.methodology import (
    graphormer_causal_analysis as graphormer_causal_analysis_module,
)
from graph_specialisation_metrics.methodology.graphormer_causal_analysis import (
    FOCUSED_CAUSAL_VERSION,
    NECESSITY_METRICS,
    PATCH_METRICS,
    _mismatch_indices,
    production_config,
    render_cached_focused_figures,
)
from graph_specialisation_metrics.methodology.graphormer_causal_plots import (
    render_focused_figure_suite,
)
from graph_specialisation_metrics.methodology.scores import (
    HeadCoordinates,
    confidence_specialists_from_draws,
    graph_local_fixed_normalization,
)


def synthetic_coordinates() -> HeadCoordinates:
    J = np.asarray([[1.20, 1.15, 0.80], [1.00, 0.10, 0.90]])
    D = np.asarray([[0.35, -0.32, 0.04], [0.00, 0.80, 0.22]])
    sem = J * (1.0 + D)
    structural = J * (1.0 - D)
    return HeadCoordinates(
        raw_semantic=sem,
        raw_structural=structural,
        semantic_mean=1.0,
        structural_mean=1.0,
        normalized_semantic=sem,
        normalized_structural=structural,
        joint_sensitivity=J,
        selectivity=D,
        active=J >= 0.20,
        estimable=True,
    )


def synthetic_gate():
    coordinates = synthetic_coordinates()
    draws = 100
    J = np.broadcast_to(coordinates.joint_sensitivity, (draws, 2, 3)).copy()
    D = np.broadcast_to(coordinates.selectivity, (draws, 2, 3)).copy()
    # One unstable head changes sign; another clears the semantic margin in 94%, not 95%.
    D[:50, 0, 2] = 0.25
    D[50:, 0, 2] = -0.25
    D[:94, 1, 2] = 0.22
    D[94:, 1, 2] = 0.05
    return confidence_specialists_from_draws(
        coordinates,
        J,
        D,
        activity_floor=0.20,
        preference_threshold=0.10,
        confidence=0.95,
        minimum_pairs=1,
    )


def test_confidence_gate_requires_joint_activity_direction_and_95_percent_support():
    gate = synthetic_gate()
    assert gate["heads"]["semantic_specialist"] == ((0, 0),)
    assert gate["heads"]["structural_specialist"] == ((0, 1),)
    assert (0, 2) in gate["heads"]["unresolved"]
    assert (1, 2) in gate["heads"]["unresolved"]
    assert (1, 1) in gate["heads"]["inactive"]
    assert gate["support"]["semantic"][1, 2] == 0.94
    assert gate["specialist_J_matching"]["pair_count"] == 1
    assert gate["semantic_null_J_matching"]["pairs"][0]["null"] == (0, 1)
    assert gate["structural_null_J_matching"]["pairs"][0]["null"] == (0, 0)

    not_estimable = confidence_specialists_from_draws(
        synthetic_coordinates(),
        np.broadcast_to(synthetic_coordinates().joint_sensitivity, (100, 2, 3)),
        np.broadcast_to(synthetic_coordinates().selectivity, (100, 2, 3)),
        minimum_pairs=3,
    )
    assert not_estimable["specialist_J_matching"]["pair_count"] == 1
    assert not_estimable["status"] == "not_estimable"


def test_activation_control_is_same_source_tier_distinct_payload_and_nearest_dose():
    def record(source, tier, payload, dose, draw):
        return SimpleNamespace(
            source=source,
            degree_gap=tier,
            payload_fingerprint=payload,
            dose=dose,
            draw=draw,
        )

    records = (
        record(0, 1, "event", 1.00, 0),
        record(0, 1, "event", 1.01, 1),  # same payload: inadmissible
        record(0, 2, "other-tier", 1.01, 2),  # wrong tier: inadmissible
        record(0, 1, "far", 1.40, 3),
        record(0, 1, "nearest", 1.15, 4),
        record(1, 1, "isolated", 1.00, 5),
    )
    mismatch, controlled = _mismatch_indices(records)
    assert mismatch[0] == 4
    assert controlled[0]
    assert mismatch[5] == -1
    assert not controlled[5]


def test_focused_score_consolidation_reuses_canonical_shards_without_diagnostics(
    monkeypatch, tmp_path
):
    from graph_specialisation_metrics.methodology import runner as runner_module

    shape = (2, 3)

    def shard(channel):
        score = np.full(shape, 2.0 if channel == "semantic" else 1.0)
        return {
            "graph_id": 0,
            "score": score,
            "event_rows": [
                {
                    "graph_id": 0,
                    "source": 0,
                    "draw": 0,
                    "score": score,
                }
            ],
            # Complete canonical shards contain these fields, but the focused
            # consolidation must neither consume nor recompute them.
            "contribution": np.ones((*shape, 2)),
            "support": np.ones(2),
            "observations": (),
            "distance_observations": (),
            "throughput": np.ones(shape),
            "attention": np.ones((*shape, 2)),
        }

    class FakeCache:
        def __init__(self):
            self.saved = []
            self.loads = []

        def load(self, stage, name, *, strict=False):
            del strict
            self.loads.append((stage, name))
            if stage == "scores/semantic" and name == "graph_000000":
                return shard("semantic")
            if stage == "scores/structural" and name == "graph_000000":
                return shard("structural")
            return None

        def save(self, stage, name, value):
            self.saved.append((stage, name, value))

        def save_audit(self, name, value):
            self.saved.append(("audit", name, value))

    cache = FakeCache()
    monkeypatch.setattr(runner_module, "_cache", lambda *_args: cache)
    monkeypatch.setattr(
        runner_module,
        "_prepare_clean_jacobians",
        lambda *_args: ({}, {"focused_cache_only": True}),
    )
    monkeypatch.setattr(
        runner_module,
        "_channel_bootstrap_policy",
        lambda *_args: SimpleNamespace(resample_source=True),
    )
    monkeypatch.setattr(
        runner_module,
        "score_heatmaps",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("focused scores must skip distance heatmaps")
        ),
    )
    monkeypatch.setattr(
        runner_module,
        "nested_percentile_interval",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("focused scores must skip canonical bootstraps")
        ),
    )

    config = production_config(
        output_dir=str(tmp_path),
        dataset_root=str(tmp_path / "pcqm"),
        cache_dir=str(tmp_path / "hf"),
        accelerator="cpu",
    )
    prepared = SimpleNamespace(
        task=SimpleNamespace(name="graphormer_pcqm4mv2", raw_score_system="mass"),
        progress=None,
    )
    plan = {
        0: {
            "semantic": {"sources": (0,), "records": ()},
            "structural": {"sources": (0,), "records": ()},
        }
    }
    result = runner_module.run_scores(
        prepared,
        config,
        plan=plan,
        focused_only=True,
    )
    assert result["focused_score_inputs_version"] == "focused-raw-score-inputs-v1"
    assert "distance_intervals" not in result["channels"]["semantic"]
    assert ("scores/semantic", "graph_000000") in cache.loads
    assert ("scores/structural", "graph_000000") in cache.loads
    assert any(
        stage == "focused/scores" and name == "raw_inputs_v1"
        for stage, name, _value in cache.saved
    )


def test_graph_local_diagnostic_uses_frozen_aggregate_normalization():
    coordinates = synthetic_coordinates()
    coordinates = HeadCoordinates(
        **{
            **coordinates.__dict__,
            "semantic_mean": 2.0,
            "structural_mean": 4.0,
        }
    )
    semantic = {10: np.full((2, 3), 4.0), 20: np.full((2, 3), 2.0)}
    structural = {10: np.full((2, 3), 4.0), 20: np.full((2, 3), 8.0)}
    result = graph_local_fixed_normalization(
        semantic, structural, coordinates, epsilon=1e-12
    )
    assert result["normalization"] == "frozen aggregate discovery channel means"
    assert np.allclose(result["selectivity"][0], 1.0 / 3.0)
    assert np.allclose(result["selectivity"][1], -1.0 / 3.0)


def _summary(metric_order, *, cells=5, nulls=2):
    count = len(metric_order)
    estimate = np.linspace(0.02, 0.18, count * cells).reshape(count, cells)
    null = np.linspace(-0.02, 0.04, count * nulls).reshape(count, nulls)
    return {
        "metric_order": tuple(metric_order),
        "cell_order": tuple(str(value) for value in range(cells)),
        "null_order": tuple(str(value) for value in range(nulls)),
        "cell_estimate": estimate,
        "cell_low": estimate - 0.01,
        "cell_high": estimate + 0.01,
        "null_estimate": null,
        "null_low": null - 0.01,
        "null_high": null + 0.01,
        "head_order": ((0, 0), (0, 1)),
        "head_estimate": np.zeros((2, 2, count)),
        "head_low": np.zeros((2, 2, count)),
        "head_high": np.zeros((2, 2, count)),
        "event_counts": {"semantic": 10, "structural": 10},
    }


def _focused_figure_inputs():
    coordinates = synthetic_coordinates()
    gate = synthetic_gate()
    gate["coordinate_interval"] = {
        "low": np.stack((coordinates.joint_sensitivity, coordinates.selectivity)),
        "high": np.stack((coordinates.joint_sensitivity, coordinates.selectivity)),
    }
    gate["molecule_diagnostic"] = {
        "semantic_fraction": np.full((2, 3), 0.5),
        "structural_fraction": np.full((2, 3), 0.25),
        "central_fraction": np.full((2, 3), 0.25),
    }
    core = {
        "version": FOCUSED_CAUSAL_VERSION,
        "target_heads": ((0, 0), (0, 1)),
        "patch": _summary(PATCH_METRICS),
        "necessity": _summary(NECESSITY_METRICS),
        "clean_ablation": {
            "J": coordinates.joint_sensitivity.reshape(-1),
            "layers": np.repeat(np.arange(2), 3),
            "prediction_movement": np.linspace(0.01, 0.08, 6),
            "prediction_movement_low": np.linspace(0.005, 0.075, 6),
            "prediction_movement_high": np.linspace(0.015, 0.085, 6),
            "spearman_rho": 0.50,
            "spearman_low": 0.20,
            "spearman_high": 0.70,
            "layer_adjusted_standardized_beta": 0.40,
            "layer_adjusted_low": 0.10,
            "layer_adjusted_high": 0.60,
            "head_count": 6,
            "graph_count": 128,
        },
        "sample_sizes": {
            "discovery_graphs": 128,
            "causal_graphs": 128,
            "clean_ablation_graphs": 128,
        },
        "control_audit": {"controlled_fraction": 0.95},
    }
    return coordinates, gate, core


def test_focused_figures_export_png_pdf_and_metadata(tmp_path):
    coordinates, gate, core = _focused_figure_inputs()
    outputs = render_focused_figure_suite(
        {"coordinates": coordinates},
        gate,
        core,
        output_dir=tmp_path,
        common_metadata={"test": True},
    )
    assert set(outputs) == {
        "specialist_gate",
        "restoration_injection",
        "necessity",
        "J_clean_ablation",
        "paper_head_ablation",
        "paper_causal_validation",
    }
    for paths in outputs.values():
        assert {Path(path).suffix for path in paths} == {".pdf", ".png"}
        for path in paths:
            assert Path(path).stat().st_size > 0
        metadata = Path(paths[0]).with_suffix(".metadata.json")
        assert metadata.stat().st_size > 0
    for key in ("paper_head_ablation", "paper_causal_validation"):
        pdf_path = next(Path(path) for path in outputs[key] if path.endswith(".pdf"))
        pdf_bytes = pdf_path.read_bytes()
        assert b"/Subtype /Type3" not in pdf_bytes
        assert b"/CIDFontType2" in pdf_bytes
        assert b"/FontFile2" in pdf_bytes
    causal_metadata = json.loads(
        (tmp_path / "02_causal_validation.metadata.json").read_text(encoding="utf-8")
    )
    assert causal_metadata["aligned_effect_axis"] == "Aligned output effect"
    assert causal_metadata["panel_order"] == [
        "Restoration",
        "Injection",
        "Role-specific necessity",
    ]
    assert causal_metadata["pdf_export"]["raster_fallback_dpi"] == 1200
    paper_manifest = json.loads(
        (tmp_path / "paper_causal_figures.json").read_text(encoding="utf-8")
    )
    assert set(paper_manifest["figures"]) == {
        "paper_head_ablation",
        "paper_causal_validation",
    }


def test_figures_phase_renders_paper_outputs_without_optional_score_cache(
    monkeypatch, tmp_path
):
    _coordinates, gate, core = _focused_figure_inputs()
    core["version"] = "graphormer-pcqm-focused-causal-v3"
    paths = graphormer_causal_analysis_module._artifact_paths(tmp_path)
    for name in ("gate", "core"):
        paths[name].parent.mkdir(parents=True, exist_ok=True)
        paths[name].touch()

    artifacts = {
        paths["gate"]: SimpleNamespace(
            value=gate,
            file_sha256="gate-sha256",
            metadata={"contract_fingerprint": "gate-contract"},
        ),
        paths["core"]: SimpleNamespace(
            value=core,
            file_sha256="core-sha256",
            metadata={"contract_fingerprint": "core-contract"},
        ),
    }
    monkeypatch.setattr(
        graphormer_causal_analysis_module,
        "load_cache_artifact_file",
        lambda path: artifacts[Path(path)],
    )

    manifest = render_cached_focused_figures(tmp_path)

    assert manifest["render_scope"] == "paper-only"
    assert manifest["score_cache"] is None
    assert (
        manifest["source_cache_analysis_version"]
        == "graphormer-pcqm-focused-causal-v3"
    )
    assert manifest["source_cache_compatibility"] == "backward-compatible"
    assert set(manifest["figures"]) == {
        "paper_head_ablation",
        "paper_causal_validation",
    }
    for stem in (
        "01_joint_sensitivity_head_ablation",
        "02_causal_validation",
    ):
        assert (paths["figures"] / f"{stem}.pdf").stat().st_size > 0
        assert (paths["figures"] / f"{stem}.png").stat().st_size > 0


def test_production_config_and_colab_freeze_128_molecule_splits():
    config = production_config(
        output_dir="/tmp/output",
        dataset_root="/tmp/pcqm",
        cache_dir="/tmp/hf",
        accelerator="cpu",
    )
    assert (
        config.sizes.discovery_graphs,
        config.sizes.causal_graphs,
        config.sizes.clean_ablation_graphs,
    ) == (128, 128, 128)
    notebook_path = (
        Path(__file__).parents[1]
        / "experiments"
        / "methodology"
        / "graphormer_pcqm4mv2_causal_colab.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", ()))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    assert 'PHASE = "figures"' in source
    assert "OUTPUT_ROOT" in source
    assert "phase=PHASE" in source
    assert "load_cache_artifact_file" in source
    assert "01_joint_sensitivity_head_ablation.pdf" in source
    assert "02_causal_validation.pdf" in source
