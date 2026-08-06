"""Cache-only paper figures shared by Graphormer, ZINC, and QM9.

The visual grammar is deliberately delegated to the approved GraphBench
population renderer.  Adapters in this module only translate the two cached
causal schemas into that renderer's plot payload; they never load a model,
dataset, checkpoint, or recompute an intervention.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .cache import atomic_json
from .figures import FigureBuilder, FigureTheme
from .graphbench_population_figures import (
    _family_necessity_mean,
    _j_matched_null_heads,
    _plot_absolute_patching,
    _plot_head_scatter,
    _value,
    population_figure_theme,
)

PAPER_ABLATION_STEM = "01_joint_sensitivity_head_ablation"
PAPER_CAUSAL_STEM = "02_causal_validation"
CHANNELS = ("semantic", "structural")


def _interval_array(interval: Any, name: str) -> np.ndarray:
    value = interval[name] if isinstance(interval, Mapping) else getattr(interval, name)
    return np.asarray(value, dtype=np.float64)


def _head_name(head: Sequence[int]) -> str:
    return f"head_L{int(head[0])}_H{int(head[1])}"


def _causal_plot_data(
    patch_estimate: np.ndarray,
    patch_low: np.ndarray,
    patch_high: np.ndarray,
    necessity_estimate: np.ndarray,
    necessity_low: np.ndarray,
    necessity_high: np.ndarray,
) -> dict[str, Any]:
    patch_estimate = np.asarray(patch_estimate, dtype=np.float64)
    patch_low = np.asarray(patch_low, dtype=np.float64)
    patch_high = np.asarray(patch_high, dtype=np.float64)
    necessity_estimate = np.asarray(necessity_estimate, dtype=np.float64)
    necessity_low = np.asarray(necessity_low, dtype=np.float64)
    necessity_high = np.asarray(necessity_high, dtype=np.float64)
    if patch_estimate.shape != (2, 2, 2):
        raise ValueError(
            "causal patch payload must be metric x head-family x donor-swap "
            f"with shape (2, 2, 2), got {patch_estimate.shape}"
        )
    if necessity_estimate.shape != (3, 2):
        raise ValueError(
            "necessity payload must be head-family x donor-swap with shape "
            f"(3, 2), got {necessity_estimate.shape}"
        )
    return {
        "absolute_patching": {
            "population": {
                "estimate": patch_estimate,
                "low": patch_low,
                "high": patch_high,
            }
        },
        "necessity": {
            "population": {
                "estimate": necessity_estimate,
                "low": necessity_low,
                "high": necessity_high,
            }
        },
    }


def _ablation_plot_data(
    *,
    seed: int,
    joint_sensitivity: Any,
    movement: Any,
    layers: Any,
    rho: float,
    rho_low: float | None,
    rho_high: float | None,
) -> dict[str, Any]:
    joint = np.asarray(joint_sensitivity, dtype=np.float64)
    impact = np.asarray(movement, dtype=np.float64).reshape(joint.shape)
    layer = np.asarray(layers, dtype=np.int64).reshape(joint.shape)
    return {
        "seed": int(seed),
        "joint_sensitivity": joint,
        "clean_ablation": impact,
        "layer": layer,
        "rho": float(rho),
        "rho_low": None if rho_low is None else float(rho_low),
        "rho_high": None if rho_high is None else float(rho_high),
    }


def canonical_paper_figure_data(
    scores: Mapping[str, Any],
    causal: Mapping[str, Any],
    *,
    seed: int,
    effect_floor: float = 1.0e-12,
) -> dict[str, Any]:
    """Translate a canonical score/causal cache pair without recomputation."""

    focused = causal.get("focused_specialists", {})
    if focused.get("status") != "estimable":
        raise ValueError(
            "paper causal panels require an estimable strongest-candidate cache; "
            f"observed status {focused.get('status')!r}"
        )
    pair_set = "strongest_candidates"
    pair_sets = list(focused.get("pair_set_order", ()))
    if pair_set not in pair_sets:
        raise KeyError(f"focused causal cache has no {pair_set!r} pair set")
    set_position = pair_sets.index(pair_set)
    metric_order = list(focused["metric_order"])
    interval = focused["interval"]

    def cells(metric: str, field: str) -> np.ndarray:
        values = _interval_array(interval, field)
        return values[set_position, metric_order.index(metric), :4].reshape(2, 2)

    patch_estimate = np.stack(
        [cells(metric, "estimate") for metric in ("restoration", "injection")]
    )
    patch_low = np.stack(
        [cells(metric, "low") for metric in ("restoration", "injection")]
    )
    patch_high = np.stack(
        [cells(metric, "high") for metric in ("restoration", "injection")]
    )
    necessity_estimate = np.full((3, 2), np.nan, dtype=np.float64)
    necessity_low = np.full((3, 2), np.nan, dtype=np.float64)
    necessity_high = np.full((3, 2), np.nan, dtype=np.float64)
    necessity_estimate[:2] = cells("necessity_fraction", "estimate")
    necessity_low[:2] = cells("necessity_fraction", "low")
    necessity_high[:2] = cells("necessity_fraction", "high")

    pairs = tuple(focused["pair_sets"][pair_set]["pairs"])
    selected = tuple(
        tuple(map(int, pair[family]))
        for family in ("semantic", "structural")
        for pair in pairs
    )
    coordinates = scores["coordinates"]
    null_match = _j_matched_null_heads(
        np.asarray(_value(coordinates, "joint_sensitivity"), dtype=np.float64),
        selected,
    )
    event_records = causal.get("event_records", {})
    for channel_position, channel in enumerate(CHANNELS):
        necessity_estimate[2, channel_position] = _family_necessity_mean(
            event_records,
            null_match["heads"],
            channel,
            effect_floor=float(effect_floor),
        )

    joint = np.asarray(
        _value(coordinates, "joint_sensitivity"), dtype=np.float64
    )
    layer = np.broadcast_to(
        np.arange(joint.shape[0], dtype=np.int64)[:, None], joint.shape
    ).copy()
    names = [
        _head_name((layer_index, head_index))
        for layer_index in range(joint.shape[0])
        for head_index in range(joint.shape[1])
    ]
    clean = causal["clean_ablation"]
    movement = np.asarray(
        [clean[name]["prediction_movement"] for name in names],
        dtype=np.float64,
    ).reshape(joint.shape)
    association = (
        causal.get("associations", {})
        .get("J_vs_clean_prediction_movement", {})
        .get("pooled", {})
    )
    rho = association.get("rho", np.nan)
    rho_low = rho_high = None
    interval_meta = clean.get("_intervals", {})
    association_order = list(interval_meta.get("association_order", ()))
    association_interval = interval_meta.get("association_interval")
    if (
        association_interval is not None
        and "J_vs_clean_prediction_movement" in association_order
    ):
        position = association_order.index("J_vs_clean_prediction_movement")
        rho_low = float(_interval_array(association_interval, "low")[position])
        rho_high = float(_interval_array(association_interval, "high")[position])

    return {
        "causal": _causal_plot_data(
            patch_estimate,
            patch_low,
            patch_high,
            necessity_estimate,
            necessity_low,
            necessity_high,
        ),
        "ablation": _ablation_plot_data(
            seed=seed,
            joint_sensitivity=joint,
            movement=movement,
            layers=layer,
            rho=float(rho),
            rho_low=rho_low,
            rho_high=rho_high,
        ),
        "pair_set": pair_set,
        "pair_count": len(pairs),
        "necessity_null": null_match,
        "source": "canonical score and causal validation caches",
    }


def graphormer_focused_paper_figure_data(
    core: Mapping[str, Any],
    gate: Mapping[str, Any],
    *,
    seed: int = 0,
) -> dict[str, Any]:
    """Translate the focused PCQM cache schema into the shared paper payload."""

    patch = core.get("patch")
    necessity = core.get("necessity")
    if patch is None or necessity is None:
        raise ValueError("focused Graphormer patch and necessity caches must be estimable")

    def summary_cells(summary: Mapping[str, Any], metric: str, field: str) -> np.ndarray:
        metric_position = tuple(summary["metric_order"]).index(metric)
        values = np.asarray(summary[field], dtype=np.float64)
        return values[metric_position, :4].reshape(2, 2)

    patch_estimate = np.stack(
        [
            summary_cells(patch, metric, "cell_estimate")
            for metric in ("R_align_adj", "I_align_adj")
        ]
    )
    patch_low = np.stack(
        [
            summary_cells(patch, metric, "cell_low")
            for metric in ("R_align_adj", "I_align_adj")
        ]
    )
    patch_high = np.stack(
        [
            summary_cells(patch, metric, "cell_high")
            for metric in ("R_align_adj", "I_align_adj")
        ]
    )
    necessity_estimate = np.full((3, 2), np.nan, dtype=np.float64)
    necessity_low = np.full((3, 2), np.nan, dtype=np.float64)
    necessity_high = np.full((3, 2), np.nan, dtype=np.float64)
    necessity_estimate[:2] = summary_cells(
        necessity, "N_fraction", "cell_estimate"
    )
    necessity_low[:2] = summary_cells(necessity, "N_fraction", "cell_low")
    necessity_high[:2] = summary_cells(necessity, "N_fraction", "cell_high")

    null_heads = tuple(
        dict.fromkeys(
            tuple(map(int, pair["null"]))
            for matching_name in (
                "semantic_null_J_matching",
                "structural_null_J_matching",
            )
            for pair in gate[matching_name].get("pairs", ())
        )
    )
    head_order = [tuple(map(int, head)) for head in necessity["head_order"]]
    null_positions = [
        head_order.index(head) for head in null_heads if head in head_order
    ]
    if null_positions:
        metric_position = tuple(necessity["metric_order"]).index("N_fraction")
        head_estimate = np.asarray(necessity["head_estimate"], dtype=np.float64)
        necessity_estimate[2] = np.mean(
            head_estimate[:, null_positions, metric_position], axis=1
        )

    clean = core["clean_ablation"]
    joint = np.asarray(clean["J"], dtype=np.float64)
    layers = np.asarray(clean["layers"], dtype=np.int64)
    return {
        "causal": _causal_plot_data(
            patch_estimate,
            patch_low,
            patch_high,
            necessity_estimate,
            necessity_low,
            necessity_high,
        ),
        "ablation": _ablation_plot_data(
            seed=seed,
            joint_sensitivity=joint,
            movement=clean["prediction_movement"],
            layers=layers,
            rho=float(clean["spearman_rho"]),
            rho_low=float(clean["spearman_low"]),
            rho_high=float(clean["spearman_high"]),
        ),
        "pair_set": "bootstrap-confidence specialists",
        "pair_count": int(gate["specialist_J_matching"]["pair_count"]),
        "necessity_null": {
            "method": "cached Graphormer J matching",
            "heads": null_heads,
        },
        "source": "focused Graphormer gate and core causal caches",
    }


def _rho_label(ablation: Mapping[str, Any]) -> str:
    label = rf"Spearman $\rho$ = {float(ablation['rho']):.2f}"
    low, high = ablation.get("rho_low"), ablation.get("rho_high")
    if low is not None and high is not None and np.isfinite((low, high)).all():
        label += rf"  [{float(low):.2f}, {float(high):.2f}]"
    return label


def render_paper_causal_figures(
    data: Mapping[str, Any],
    *,
    output_dir: str | Path,
    task_name: str,
    seed: int,
    common_metadata: Mapping[str, Any] | None = None,
    theme: FigureTheme | None = None,
) -> dict[str, list[str]]:
    """Write the two independent paper panels from cached payloads only."""

    paper_theme = theme or population_figure_theme()
    output_path = Path(output_dir)
    builder = FigureBuilder(
        output_path,
        theme=paper_theme,
        common_metadata={
            "task": str(task_name),
            "seed": int(seed),
            "cache_only": True,
            "visual_reference": "GraphBench population causal validation",
            **dict(common_metadata or {}),
        },
        preserve_canvas=True,
    )
    outputs: dict[str, list[str]] = {}

    ablation = data["ablation"]
    scatter_data = {
        "seeds": np.asarray([int(seed)], dtype=np.int64),
        "heads": [
            {
                "seed": int(seed),
                "layer": ablation["layer"],
                "joint_sensitivity": ablation["joint_sensitivity"],
                "clean_ablation": ablation["clean_ablation"],
            }
        ],
    }
    figure, axis = _plot_head_scatter(
        scatter_data,
        paper_theme,
        x_name="joint_sensitivity",
        y_name="clean_ablation",
        xlabel=r"Joint sensitivity, $J$",
        ylabel="Head-ablation impact",
        title="Joint sensitivity and head-ablation impact",
        statistic=_rho_label(ablation),
    )
    axis.set_xlim(left=0.0)
    axis.set_ylim(bottom=0.0)
    paths = builder.save(
        PAPER_ABLATION_STEM,
        figure,
        axis,
        metadata={
            "figure_role": "standalone paper head-ablation validation",
            "association": {
                "statistic": "Spearman rho",
                "rho": ablation["rho"],
                "low": ablation.get("rho_low"),
                "high": ablation.get("rho_high"),
            },
            "x_axis": "Joint sensitivity J",
            "y_axis": "Head-ablation impact",
            "point_color": "transformer layer",
        },
    )
    outputs["paper_head_ablation"] = [str(path) for path in paths]

    figure, axes = _plot_absolute_patching(
        data["causal"],
        paper_theme,
        legend_title="Mean and 95% bootstrap CI",
    )
    paths = builder.save(
        PAPER_CAUSAL_STEM,
        figure,
        axes,
        metadata={
            "figure_role": "standalone paper causal validation panels",
            "panel_order": (
                "Restoration",
                "Injection",
                "Role-specific necessity",
            ),
            "aligned_effect_axis": "Aligned output effect",
            "pair_set": data.get("pair_set"),
            "pair_count": data.get("pair_count"),
            "necessity_null": data.get("necessity_null"),
            "cache_source": data.get("source"),
        },
    )
    outputs["paper_causal_validation"] = [str(path) for path in paths]

    atomic_json(
        output_path / "paper_causal_figures.json",
        {
            "task": str(task_name),
            "seed": int(seed),
            "cache_only": True,
            "figures": outputs,
            "paper_composition": (
                "Keep the two PDFs separate here; combine them only in the final "
                "paper layout."
            ),
        },
    )
    return outputs


def render_canonical_paper_causal_figures(
    scores: Mapping[str, Any],
    causal: Mapping[str, Any],
    *,
    output_dir: str | Path,
    task_name: str,
    seed: int,
    common_metadata: Mapping[str, Any] | None = None,
) -> dict[str, list[str]]:
    data = canonical_paper_figure_data(scores, causal, seed=seed)
    return render_paper_causal_figures(
        data,
        output_dir=output_dir,
        task_name=task_name,
        seed=seed,
        common_metadata=common_metadata,
    )


__all__ = [
    "PAPER_ABLATION_STEM",
    "PAPER_CAUSAL_STEM",
    "canonical_paper_figure_data",
    "graphormer_focused_paper_figure_data",
    "render_canonical_paper_causal_figures",
    "render_paper_causal_figures",
]
