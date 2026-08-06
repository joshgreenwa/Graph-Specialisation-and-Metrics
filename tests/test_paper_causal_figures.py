from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from graph_specialisation_metrics.methodology.bootstrap import Interval
from graph_specialisation_metrics.methodology.graphbench_population_figures import (
    ABLATION_SCATTER_ALPHA,
    _plot_head_scatter,
    population_figure_theme,
)
from graph_specialisation_metrics.methodology.paper_causal_figures import (
    canonical_paper_figure_data,
    derive_legacy_focused_specialists,
    render_canonical_paper_causal_figures,
)
from graph_specialisation_metrics.methodology.protocol import (
    FamilyPolicy,
    MethodologyConfig,
)
from graph_specialisation_metrics.methodology.scores import HeadCoordinates


def _canonical_cache_values():
    raw_semantic = np.asarray(((0.9, 0.4), (0.7, 0.5)))
    raw_structural = np.asarray(((0.3, 0.8), (0.6, 0.55)))
    semantic = raw_semantic / np.mean(raw_semantic)
    structural = raw_structural / np.mean(raw_structural)
    joint = 0.5 * (semantic + structural)
    selectivity = (semantic - structural) / (semantic + structural)
    coordinates = HeadCoordinates(
        raw_semantic=raw_semantic,
        raw_structural=raw_structural,
        semantic_mean=float(np.mean(raw_semantic)),
        structural_mean=float(np.mean(raw_structural)),
        normalized_semantic=semantic,
        normalized_structural=structural,
        joint_sensitivity=joint,
        selectivity=selectivity,
        active=np.ones_like(joint, dtype=bool),
        estimable=True,
    )
    estimates = np.zeros((1, 4, 5), dtype=np.float64)
    estimates[0, 0] = (0.20, 0.12, 0.10, 0.18, 0.16)
    estimates[0, 1] = (0.16, 0.09, 0.08, 0.14, 0.13)
    estimates[0, 2] = (0.48, 0.32, 0.28, 0.44, 0.32)
    estimates[0, 3] = (0.52, 0.35, 0.30, 0.46, 0.33)
    focused_interval = Interval(
        estimate=estimates,
        low=estimates - 0.02,
        high=estimates + 0.02,
        replicates=2_000,
        rng_seed=17_071,
        resampled_levels=("graph", "source", "donor"),
    )
    association_interval = Interval(
        estimate=np.asarray((0.62, 0.20)),
        low=np.asarray((0.40, -0.05)),
        high=np.asarray((0.78, 0.42)),
        replicates=2_000,
        rng_seed=17_071,
        resampled_levels=("graph",),
    )
    event_records = {}
    for head_position, name in enumerate(
        ("head_L0_H0", "head_L0_H1", "head_L1_H0", "head_L1_H1")
    ):
        event_records[name] = {
            channel: [
                {
                    "graph": graph,
                    "source": 0,
                    "donor": 0,
                    "necessity": 0.04 + 0.01 * head_position,
                    "event_effect": 0.40 + 0.01 * graph,
                }
                for graph in (0, 1)
            ]
            for channel in ("semantic", "structural")
        }
    clean = {
        "head_L0_H0": {"prediction_movement": 0.42},
        "head_L0_H1": {"prediction_movement": 0.18},
        "head_L1_H0": {"prediction_movement": 0.31},
        "head_L1_H1": {"prediction_movement": 0.24},
        "_intervals": {
            "association_order": (
                "J_vs_clean_prediction_movement",
                "J_vs_clean_loss_change",
            ),
            "association_interval": association_interval,
        },
    }
    scores = {"coordinates": coordinates}
    causal = {
        "event_records": event_records,
        "clean_ablation": clean,
        "associations": {
            "J_vs_clean_prediction_movement": {
                "pooled": {"rho": 0.62, "n": 4}
            }
        },
        "focused_specialists": {
            "status": "estimable",
            "pair_set_order": ("strongest_candidates",),
            "pair_sets": {
                "strongest_candidates": {
                    "pair_count": 1,
                    "pairs": (
                        {"semantic": (0, 0), "structural": (0, 1)},
                    ),
                }
            },
            "metric_order": (
                "restoration",
                "injection",
                "necessity_fraction",
                "gross_necessity_fraction",
            ),
            "interval": focused_interval,
        },
    }
    return scores, causal


def test_legacy_causal_cache_materializes_missing_focused_summary(tmp_path):
    scores, causal = _canonical_cache_values()
    causal = dict(causal)
    causal.pop("focused_specialists")
    selectivity = np.asarray(scores["coordinates"].selectivity)
    score_estimate = np.zeros((6, *selectivity.shape), dtype=np.float64)
    score_estimate[5] = selectivity
    scores = {
        **scores,
        "intervals": Interval(
            estimate=score_estimate,
            low=score_estimate - 0.02,
            high=score_estimate + 0.02,
            replicates=2_000,
            rng_seed=17_071,
            resampled_levels=("graph", "source", "donor"),
        ),
    }
    for head_position, channels in enumerate(causal["event_records"].values()):
        for channel_position, rows in enumerate(channels.values()):
            for row in rows:
                row.update(
                    {
                        "P_gross_matched": 0.40 + 0.01 * head_position,
                        "gross_necessity": 0.08 + 0.01 * head_position,
                        "R_align_adjusted": (
                            0.16 + 0.01 * head_position + 0.01 * channel_position
                        ),
                        "I_align_adjusted": (
                            0.12 + 0.01 * head_position + 0.01 * channel_position
                        ),
                    }
                )
    config = MethodologyConfig(
        output_dir=str(tmp_path),
        tasks=("zinc",),
        train_seeds=(42,),
        phases=("figures",),
        accelerator="cpu",
        families=FamilyPolicy(specialist_minimum_pairs=1),
    )

    focused = derive_legacy_focused_specialists(scores, causal, config=config)

    assert focused["status"] == "estimable"
    assert focused["pair_sets"]["strongest_candidates"]["pair_count"] == 1
    assert np.isfinite(focused["interval"].estimate).all()


def test_head_ablation_scatter_uses_opaque_points_and_boxed_black_statistic():
    import matplotlib.pyplot as plt

    data = {
        "seeds": np.asarray((42,)),
        "heads": (
            {
                "joint_sensitivity": np.asarray((0.4, 0.8, 1.2)),
                "clean_ablation": np.asarray((0.2, 0.5, 0.9)),
                "layer": np.asarray((0, 1, 2)),
            },
        ),
    }
    figure, axis = _plot_head_scatter(
        data,
        population_figure_theme(),
        x_name="joint_sensitivity",
        y_name="clean_ablation",
        xlabel=r"Joint sensitivity, $J$",
        ylabel="Head-ablation impact",
        title="Joint sensitivity and head-ablation impact",
        statistic=r"Spearman $\rho$ = 0.53  [0.48, 0.56]",
    )
    try:
        assert axis.collections[0].get_alpha() == ABLATION_SCATTER_ALPHA
        statistic = axis.texts[-1]
        assert statistic.get_color() == "black"
        box = statistic.get_bbox_patch()
        assert np.allclose(box.get_facecolor()[:3], (1.0, 1.0, 1.0))
        assert box.get_alpha() == 0.96
    finally:
        plt.close(figure)


def test_canonical_cache_adapter_writes_two_graphbench_matched_paper_pdfs(tmp_path):
    scores, causal = _canonical_cache_values()
    data = canonical_paper_figure_data(scores, causal, seed=42)
    assert data["causal"]["absolute_patching"]["population"]["estimate"].shape == (
        2,
        2,
        2,
    )
    assert data["causal"]["necessity"]["population"]["estimate"].shape == (3, 2)
    assert data["ablation"]["rho"] == 0.62
    assert data["ablation"]["rho_low"] == 0.40
    assert data["ablation"]["rho_high"] == 0.78

    outputs = render_canonical_paper_causal_figures(
        scores,
        causal,
        output_dir=tmp_path,
        task_name="zinc",
        seed=42,
        common_metadata={"test": True},
    )
    assert set(outputs) == {"paper_head_ablation", "paper_causal_validation"}
    for paths in outputs.values():
        assert {Path(path).suffix for path in paths} == {".pdf", ".png"}
        pdf = next(Path(path) for path in paths if path.endswith(".pdf"))
        pdf_bytes = pdf.read_bytes()
        assert b"/Subtype /Type3" not in pdf_bytes
        assert b"/CIDFontType2" in pdf_bytes
        assert b"/FontFile2" in pdf_bytes
    ablation_metadata = json.loads(
        (tmp_path / "01_joint_sensitivity_head_ablation.metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert ablation_metadata["x_axis"] == "Joint sensitivity J"
    assert ablation_metadata["y_axis"] == "Head-ablation impact"
    assert ablation_metadata["association"]["rho"] == 0.62
    assert ablation_metadata["pdf_export"]["raster_fallback_dpi"] == 1200
