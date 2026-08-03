"""Paper figures for the Graphormer causal head-population analysis."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .figures import FigureBuilder, FigureTheme, publication_style

SEMANTIC = "#C44E52"
STRUCTURAL = "#4C72B0"
NULL = "#777777"


def _theme() -> FigureTheme:
    return FigureTheme(
        width=4.2,
        height=3.55,
        dpi=600,
        font_size=9.8,
        label_size=10.4,
        title_size=10.8,
        tick_size=9.0,
        marker_size=34.0,
        line_width=1.25,
        grid_alpha=0.14,
        formats=("pdf", "png"),
    )


def _style_axis(ax: Any) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.14, linewidth=0.55)
    ax.set_axisbelow(True)


def _colour_intervention_ticks(ax: Any) -> None:
    labels = ax.get_xticklabels()
    if len(labels) >= 2:
        labels[0].set_color(SEMANTIC)
        labels[1].set_color(STRUCTURAL)


def _family_handles(*, include_null: bool):
    from matplotlib.lines import Line2D

    values = [
        ("Semantic-scoring heads", SEMANTIC),
        ("Structural-scoring heads", STRUCTURAL),
    ]
    if include_null:
        values.append((r"$J$-matched null heads", NULL))
    return [
        Line2D(
            [0],
            [0],
            color=colour,
            marker="o",
            linestyle="none",
            markeredgecolor="white",
            markeredgewidth=0.55,
            label=label,
        )
        for label, colour in values
    ]


def _population_marks(ax: Any, endpoint: Mapping[str, Any], *, include_null: bool):
    estimates = np.asarray(endpoint["estimate"], dtype=np.float64)
    lows = np.asarray(endpoint["low"], dtype=np.float64)
    highs = np.asarray(endpoint["high"], dtype=np.float64)
    colours = [SEMANTIC, STRUCTURAL]
    offsets = [-0.13, 0.13]
    if include_null:
        colours.append(NULL)
        offsets = [-0.18, 0.0, 0.18]
    x = np.arange(2, dtype=np.float64)
    for family, (colour, offset) in enumerate(zip(colours, offsets)):
        estimate = estimates[family]
        low = lows[family]
        high = highs[family]
        ax.errorbar(
            x + offset,
            estimate,
            yerr=np.maximum(0.0, np.vstack((estimate - low, high - estimate))),
            fmt="o",
            color=colour,
            markerfacecolor=colour,
            markeredgecolor="white",
            markeredgewidth=0.55,
            markersize=6.2,
            linewidth=1.25,
            capsize=3.0,
            capthick=1.0,
            linestyle="none",
            zorder=5,
        )
    ax.set_xlim(-0.30, 1.30)


def plot_population_causal_tests(core: Mapping[str, Any]):
    """Restoration, injection, and necessity in one compact primary figure."""

    import matplotlib.pyplot as plt

    theme = _theme()
    with publication_style(theme):
        fig, axes = plt.subplots(1, 3, figsize=(theme.width * 3.0, theme.height))
        for ax, endpoint_name, title in zip(
            axes[:2],
            ("restoration", "injection"),
            (
                "Clean head state in an intervened graph",
                "Intervened head state in a clean graph",
            ),
        ):
            _population_marks(ax, core[endpoint_name], include_null=False)
            ax.axhline(0.0, color="#9A9A9A", linewidth=0.75)
            ax.set_xticks(
                np.arange(2),
                ("Semantic intervention", "Structural intervention"),
            )
            _colour_intervention_ticks(ax)
            ax.set_title(title)
            _style_axis(ax)
        axes[0].set_ylabel("Output effect beyond mismatch control")

        _population_marks(axes[2], core["necessity"], include_null=True)
        axes[2].axhline(0.0, color="#9A9A9A", linewidth=0.75)
        axes[2].set_xticks(
            np.arange(2),
            ("Semantic intervention", "Structural intervention"),
        )
        _colour_intervention_ticks(axes[2])
        axes[2].set_title("Head necessity")
        axes[2].set_ylabel("Intervention effect removed (fraction)")
        _style_axis(axes[2])

        pair_count = len(core["population_gate"]["specialist_pairs"])
        null_count = len(core["population_gate"]["null_pairs"])
        legend = fig.legend(
            handles=_family_handles(include_null=True),
            frameon=False,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=3,
            columnspacing=1.2,
            handletextpad=0.45,
            borderaxespad=0,
            title=(
                "Mean and 95% paired bootstrap CI over molecules, donor events, "
                f"and heads  ({pair_count} specialist pairs; {null_count} null heads)"
            ),
        )
        legend.get_title().set_fontsize(theme.tick_size)
        fig.subplots_adjust(
            bottom=0.28,
            left=0.075,
            right=0.978,
            top=0.88,
            wspace=0.34,
        )
    return fig, axes


def plot_population_selection(scores: Mapping[str, Any], gate: Mapping[str, Any]):
    """Discovery-only audit of the selected and matched populations."""

    import matplotlib.pyplot as plt

    coordinates = scores["coordinates"]
    D = np.asarray(coordinates.selectivity, dtype=np.float64)
    J = np.asarray(coordinates.joint_sensitivity, dtype=np.float64)
    active = np.asarray(coordinates.active, dtype=bool)
    semantic = {tuple(head) for head in gate["heads"]["semantic"]}
    structural = {tuple(head) for head in gate["heads"]["structural"]}
    null = {tuple(head) for head in gate["heads"]["j_matched_null"]}

    def values(items: set[tuple[int, int]], array: np.ndarray) -> list[float]:
        return [float(array[item]) for item in sorted(items)]

    theme = _theme()
    with publication_style(theme):
        fig, ax = plt.subplots(figsize=(theme.width * 1.35, theme.height))
        background = active & np.isfinite(D) & np.isfinite(J)
        ax.scatter(
            D[background],
            J[background],
            s=18,
            color="#D5D5D5",
            edgecolor="none",
            alpha=0.72,
            label="Other active heads",
            zorder=1,
        )
        for pairs, colour, marker, label in (
            (semantic, SEMANTIC, "o", "Semantic-scoring heads"),
            (structural, STRUCTURAL, "s", "Structural-scoring heads"),
            (null, NULL, "D", r"$J$-matched null heads"),
        ):
            ax.scatter(
                values(pairs, D),
                values(pairs, J),
                s=42,
                color=colour,
                marker=marker,
                edgecolor="white",
                linewidth=0.55,
                label=f"{label} (n={len(pairs)})",
                zorder=3,
            )
        threshold = float(gate["preference_threshold"])
        ax.axvspan(-threshold, threshold, color=NULL, alpha=0.07, linewidth=0)
        ax.axvline(-threshold, color="#A0A0A0", linestyle=":", linewidth=0.8)
        ax.axvline(threshold, color="#A0A0A0", linestyle=":", linewidth=0.8)
        ax.axhline(float(gate["activity_floor"]), color="#888888", linestyle="--", linewidth=0.8)
        balance = gate["matching_balance"]
        ax.text(
            0.02,
            0.98,
            (
                r"specialist $J$ SMD="
                f"{balance['semantic_vs_structural']['standardized_J_difference']:+.2f}\n"
                r"specialist-null $J$ SMD="
                f"{balance['specialists_vs_null']['standardized_J_difference']:+.2f}"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=theme.tick_size,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86},
        )
        ax.set_xlabel(
            r"Relative selectivity $D_{rel}$  (structural $\leftarrow$ 0 $\rightarrow$ semantic)"
        )
        ax.set_ylabel(r"Joint sensitivity $J$")
        ax.set_title("Discovery-only causal head populations")
        ax.legend(frameon=False, fontsize=theme.tick_size, loc="lower right")
        _style_axis(ax)
        fig.tight_layout()
    return fig, ax


def plot_J_ablation(clean: Mapping[str, Any]):
    """Raw and layer-adjusted views of J versus clean head ablation."""

    import matplotlib as mpl
    import matplotlib.pyplot as plt

    J = np.asarray(clean["J"], dtype=np.float64)
    movement = np.asarray(clean["prediction_movement"], dtype=np.float64)
    low = np.asarray(clean["prediction_movement_low"], dtype=np.float64)
    high = np.asarray(clean["prediction_movement_high"], dtype=np.float64)
    layers = np.asarray(clean["layers"], dtype=np.int64)
    partial = clean["layer_adjusted_partial"]
    x_partial = np.asarray(partial["x_residual"], dtype=np.float64)
    y_partial = np.asarray(partial["y_residual"], dtype=np.float64)
    partial_layers = np.asarray(partial["layers"], dtype=np.int64)
    maximum_layer = int(np.max(layers))
    cmap = plt.get_cmap("viridis", maximum_layer + 1)
    norm = mpl.colors.BoundaryNorm(np.arange(-0.5, maximum_layer + 1.5, 1.0), cmap.N)
    theme = _theme()
    with publication_style(theme):
        fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.25))
        axes[0].errorbar(
            J,
            movement,
            yerr=np.maximum(0.0, np.vstack((movement - low, high - movement))),
            fmt="none",
            ecolor="#777777",
            alpha=0.15,
            linewidth=0.42,
            zorder=1,
        )
        axes[0].scatter(
            J,
            movement,
            c=layers,
            cmap=cmap,
            norm=norm,
            s=22,
            edgecolor="white",
            linewidth=0.3,
            alpha=0.88,
            zorder=2,
        )
        axes[0].set_xlabel(r"Joint sensitivity $J$")
        axes[0].set_ylabel(r"Clean head-ablation movement $\|z-z_{-h}\|_2$")
        axes[0].set_title(
            "All-head association\n"
            f"Spearman $\\rho$={clean['spearman_rho']:.2f} "
            f"[{clean['spearman_low']:.2f}, {clean['spearman_high']:.2f}]"
        )

        axes[1].scatter(
            x_partial,
            y_partial,
            c=partial_layers,
            cmap=cmap,
            norm=norm,
            s=22,
            edgecolor="white",
            linewidth=0.3,
            alpha=0.88,
            zorder=2,
        )
        extent = np.asarray([float(np.min(x_partial)), float(np.max(x_partial))], dtype=np.float64)
        axes[1].plot(
            extent,
            float(partial["beta"]) * extent,
            color="#333333",
            linewidth=1.25,
            zorder=3,
        )
        axes[1].axhline(0, color="#A0A0A0", linewidth=0.65)
        axes[1].axvline(0, color="#A0A0A0", linewidth=0.65)
        axes[1].set_xlabel("Layer-adjusted joint sensitivity (SD)")
        axes[1].set_ylabel("Layer-adjusted ablation impact (SD)")
        axes[1].set_title(
            "Within-layer association\n"
            f"standardized $\\beta$={clean['layer_adjusted_standardized_beta']:.2f} "
            f"[{clean['layer_adjusted_low']:.2f}, {clean['layer_adjusted_high']:.2f}]"
        )
        for ax in axes:
            _style_axis(ax)

        colorbar_ax = fig.add_axes((0.925, 0.23, 0.018, 0.59))
        colorbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=colorbar_ax)
        colorbar.set_label("Layer")
        colorbar.set_ticks(np.arange(maximum_layer + 1))
        fig.suptitle(
            "Joint sensitivity predicts clean-input head importance\n"
            f"PCQM4Mv2 ({clean['head_count']} heads; "
            f"{clean['graph_count']} held-out molecules)",
            fontsize=theme.title_size + 0.8,
        )
        fig.text(
            0.48,
            0.012,
            (
                r"$\beta$ is the partial slope after global SD scaling and layer fixed effects; "
                "intervals resample held-out molecules."
            ),
            ha="center",
            va="bottom",
            fontsize=theme.tick_size,
            color="#4A4A4A",
        )
        fig.subplots_adjust(
            bottom=0.23,
            left=0.085,
            right=0.90,
            top=0.78,
            wspace=0.32,
        )
    return fig, axes


def render_population_figure_suite(
    scores: Mapping[str, Any],
    gate: Mapping[str, Any],
    core: Mapping[str, Any],
    *,
    output_dir: str | Path,
    common_metadata: Mapping[str, Any],
) -> dict[str, list[str]]:
    theme = _theme()
    builder = FigureBuilder(
        Path(output_dir),
        theme=theme,
        common_metadata=dict(common_metadata),
        preserve_canvas=True,
    )
    outputs: dict[str, list[str]] = {}

    figure, axes = plot_population_causal_tests(core)
    paths = builder.save(
        "01_population_restoration_injection_necessity",
        figure,
        axes,
        metadata={
            "figure_role": "three primary causal tests",
            "restoration": core["restoration"],
            "injection": core["injection"],
            "necessity": core["necessity"],
            "matching_balance": gate["matching_balance"],
        },
    )
    outputs["population_causal_tests"] = [str(path) for path in paths]

    figure, axes = plot_J_ablation(core["clean_ablation"])
    paths = builder.save(
        "02_J_vs_clean_ablation",
        figure,
        axes,
        metadata={
            "figure_role": "raw and layer-adjusted J versus clean ablation",
            "clean_ablation": core["clean_ablation"],
        },
    )
    outputs["J_vs_clean_ablation"] = [str(path) for path in paths]

    figure, ax = plot_population_selection(scores, gate)
    paths = builder.save(
        "S01_population_head_selection",
        figure,
        ax,
        metadata={
            "figure_role": "discovery-only selection and matching audit",
            "population_gate": gate,
        },
    )
    outputs["population_selection"] = [str(path) for path in paths]
    return outputs


__all__ = [
    "plot_J_ablation",
    "plot_population_causal_tests",
    "plot_population_selection",
    "render_population_figure_suite",
]
