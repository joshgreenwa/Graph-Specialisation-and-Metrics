"""Publication figures for the focused PCQM4Mv2 Graphormer causal tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .figures import FigureBuilder, FigureTheme, publication_style


SEMANTIC = "#C44E52"
STRUCTURAL = "#4C72B0"
GENERALIST = "#7A7A7A"
UNRESOLVED = "#D7D7D7"
INACTIVE = "#F2F2F2"


def _as_heads(values: Sequence[Sequence[int]]) -> set[tuple[int, int]]:
    return {(int(value[0]), int(value[1])) for value in values}


def _head_category(gate: Mapping[str, Any], shape: tuple[int, int]):
    semantic = _as_heads(gate["heads"]["semantic_specialist"])
    structural = _as_heads(gate["heads"]["structural_specialist"])
    generalist = _as_heads(gate["heads"]["persistent_generalist"])
    inactive = _as_heads(gate["heads"]["inactive"])
    codes = np.full(shape, 3, dtype=np.int64)
    labels = np.full(shape, "unresolved", dtype=object)
    for layer in range(shape[0]):
        for head in range(shape[1]):
            item = (layer, head)
            if item in semantic:
                codes[item], labels[item] = 0, "semantic specialist"
            elif item in structural:
                codes[item], labels[item] = 1, "structural specialist"
            elif item in generalist:
                codes[item], labels[item] = 2, "confidence-supported generalist"
            elif item in inactive:
                codes[item], labels[item] = 4, "inactive"
    return codes, labels


def plot_specialist_gate(scores: Mapping[str, Any], gate: Mapping[str, Any]):
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.lines import Line2D

    coordinates = scores["coordinates"]
    D = np.asarray(coordinates.selectivity, dtype=np.float64)
    J = np.asarray(coordinates.joint_sensitivity, dtype=np.float64)
    shape = tuple(D.shape)
    codes, labels = _head_category(gate, shape)
    colours = (SEMANTIC, STRUCTURAL, GENERALIST, UNRESOLVED, INACTIVE)
    cmap = ListedColormap(colours)
    norm = BoundaryNorm(np.arange(-0.5, 5.5, 1), cmap.N)

    with publication_style(FigureTheme(dpi=600)):
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(13.2, 5.4),
            constrained_layout=True,
            gridspec_kw={"width_ratios": (1.45, 1.0)},
        )
        image = axes[0].imshow(codes, cmap=cmap, norm=norm, aspect="auto")
        del image
        axes[0].set_title("Bootstrap-confidence specialist gate")
        axes[0].set_xlabel("Attention head")
        axes[0].set_ylabel("Layer")
        axes[0].set_xticks(np.arange(shape[1]))
        axes[0].set_yticks(np.arange(shape[0]))
        axes[0].tick_params(axis="x", labelsize=7)
        axes[0].set_xticks(np.arange(-0.5, shape[1], 1), minor=True)
        axes[0].set_yticks(np.arange(-0.5, shape[0], 1), minor=True)
        axes[0].grid(which="minor", color="white", linewidth=0.35, alpha=0.8)
        axes[0].tick_params(which="minor", bottom=False, left=False)

        for code, label, colour in zip(
            range(5),
            (
                "Semantic specialist",
                "Structural specialist",
                "Confidence-supported generalist",
                "Unresolved",
                "Inactive",
            ),
            colours,
        ):
            selected = codes.reshape(-1) == code
            axes[1].scatter(
                D.reshape(-1)[selected],
                J.reshape(-1)[selected],
                s=31 if code < 2 else 22,
                c=colour,
                edgecolors="black" if code < 2 else "none",
                linewidths=0.45,
                alpha=1.0 if code < 3 else 0.75,
                label=label,
                zorder=3 if code < 2 else 2,
            )
        threshold = float(gate["preference_threshold"])
        floor = float(gate["activity_floor"])
        axes[1].axvline(threshold, color=SEMANTIC, linestyle="--", linewidth=1.0)
        axes[1].axvline(-threshold, color=STRUCTURAL, linestyle="--", linewidth=1.0)
        axes[1].axhline(floor, color="black", linestyle=":", linewidth=1.0)
        for pair in gate["specialist_J_matching"]["pairs"]:
            left = tuple(pair["semantic"])
            right = tuple(pair["structural"])
            axes[1].plot(
                (D[left], D[right]),
                (J[left], J[right]),
                color="#555555",
                alpha=0.45,
                linewidth=0.7,
                zorder=1,
            )
        null_heads: set[tuple[int, int]] = set()
        for matching, colour in (
            (gate["semantic_null_J_matching"], SEMANTIC),
            (gate["structural_null_J_matching"], STRUCTURAL),
        ):
            for pair in matching["pairs"]:
                specialist = tuple(pair["specialist"])
                null = tuple(pair["null"])
                null_heads.add(null)
                axes[1].plot(
                    (D[specialist], D[null]),
                    (J[specialist], J[null]),
                    color=colour,
                    alpha=0.28,
                    linewidth=0.7,
                    linestyle=":",
                    zorder=1,
                )
        if null_heads:
            ordered_nulls = sorted(null_heads)
            axes[1].scatter(
                [D[head] for head in ordered_nulls],
                [J[head] for head in ordered_nulls],
                marker="x",
                s=48,
                color="#111111",
                linewidths=0.9,
                label=r"Opposite-sign $J$ null",
                zorder=5,
            )
        axes[1].set_title("Frozen labels and $J$ matches")
        axes[1].set_xlabel(
            r"Selectivity $D_{rel}$  (structural $\leftarrow$ 0 $\rightarrow$ semantic)"
        )
        axes[1].set_ylabel(r"Joint sensitivity $J$")
        axes[1].grid(alpha=0.18, linewidth=0.5)
        axes[1].legend(frameon=False, fontsize=8, loc="best")
        axes[1].text(
            0.02,
            0.02,
            (
                f"semantic={len(gate['heads']['semantic_specialist'])}, "
                f"structural={len(gate['heads']['structural_specialist'])}\n"
                f"matched pairs={gate['specialist_J_matching']['pair_count']}"
            ),
            transform=axes[1].transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82},
        )
        handles = [
            Line2D([0], [0], marker="s", color="none", markerfacecolor=colour,
                   markeredgecolor="#666666", markersize=7, label=label)
            for colour, label in zip(
                colours,
                (
                    "Semantic specialist",
                    "Structural specialist",
                    "Confidence-supported generalist",
                    "Unresolved",
                    "Inactive",
                ),
            )
        ]
        axes[0].legend(
            handles=handles,
            frameon=False,
            fontsize=8,
            ncol=2,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.10),
        )
        fig.suptitle(
            "Heads are labelled only when ≥95% of discovery bootstrap draws clear "
            "$J$ and directional $D_{rel}$ thresholds",
            fontsize=12,
        )
    return fig, axes, labels


def _metric_position(summary: Mapping[str, Any], name: str) -> int:
    return tuple(summary["metric_order"]).index(name)


def _errorbar_cells(ax, summary, metric, *, title, ylabel, gross_components=None):
    index = _metric_position(summary, metric)
    estimate = np.asarray(summary["cell_estimate"])[index, :4]
    low = np.asarray(summary["cell_low"])[index, :4]
    high = np.asarray(summary["cell_high"])[index, :4]
    x = np.arange(4)
    colours = (SEMANTIC, SEMANTIC, STRUCTURAL, STRUCTURAL)
    markers = ("o", "s", "o", "s")
    for position in range(4):
        ax.errorbar(
            x[position],
            estimate[position],
            yerr=np.asarray(
                [[max(0.0, estimate[position] - low[position])],
                 [max(0.0, high[position] - estimate[position])]]
            ),
            fmt=markers[position],
            color=colours[position],
            markeredgecolor="white",
            markeredgewidth=0.6,
            markersize=7,
            capsize=3,
            linewidth=1.1,
            zorder=4,
        )
    if gross_components is not None:
        matched_name, null_name = gross_components
        matched = np.asarray(summary["cell_estimate"])[
            _metric_position(summary, matched_name), :4
        ]
        null = np.asarray(summary["cell_estimate"])[
            _metric_position(summary, null_name), :4
        ]
        ax.scatter(x - 0.08, matched, marker="^", s=23, color="#777777", alpha=0.55,
                   label="Matched raw")
        ax.scatter(x + 0.08, null, marker="v", s=23, facecolor="white",
                   edgecolor="#777777", alpha=0.75, label="Alternative-donor raw")
    interaction = float(np.asarray(summary["cell_estimate"])[index, 4])
    interaction_low = float(np.asarray(summary["cell_low"])[index, 4])
    interaction_high = float(np.asarray(summary["cell_high"])[index, 4])
    null_values = np.asarray(summary["null_estimate"])[index]
    ax.text(
        0.02,
        0.98,
        (
            f"double difference {interaction:+.3f} "
            f"[{interaction_low:+.3f}, {interaction_high:+.3f}]\n"
            f"vs $J$-null: sem {null_values[0]:+.3f}; str {null_values[1]:+.3f}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
    )
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
    ax.set_xticks(
        x,
        ("Sem H\nSem E", "Sem H\nStr E", "Str H\nSem E", "Str H\nStr E"),
    )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.18, linewidth=0.5)


def plot_restoration_injection(summary: Mapping[str, Any] | None):
    import matplotlib.pyplot as plt

    with publication_style(FigureTheme(dpi=600)):
        fig, axes = plt.subplots(
            2, 2, figsize=(11.8, 8.0), constrained_layout=True
        )
        if summary is None:
            for ax in axes.reshape(-1):
                ax.axis("off")
                ax.text(0.5, 0.5, "Not estimable: no reliable matched specialists",
                        ha="center", va="center")
            return fig, axes
        _errorbar_cells(
            axes[0, 0],
            summary,
            "R_gross_adj",
            title="Restoration — gross output movement",
            ylabel="Alternative-donor-adjusted $z$ movement",
            gross_components=("R_gross_matched", "R_gross_null"),
        )
        _errorbar_cells(
            axes[0, 1],
            summary,
            "I_gross_adj",
            title="Injection — gross output movement",
            ylabel="Alternative-donor-adjusted $z$ movement",
            gross_components=("I_gross_matched", "I_gross_null"),
        )
        _errorbar_cells(
            axes[1, 0],
            summary,
            "R_align_adj",
            title="Restoration — movement toward clean",
            ylabel="Adjusted direction-aligned movement",
            gross_components=("R_align_matched", "R_align_null"),
        )
        _errorbar_cells(
            axes[1, 1],
            summary,
            "I_align_adj",
            title="Injection — movement toward intervention",
            ylabel="Adjusted direction-aligned movement",
            gross_components=("I_align_matched", "I_align_null"),
        )
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, frameon=False, loc="lower center", ncol=2)
        fig.suptitle(
            "Bidirectional head-output patching on held-out donor events",
            fontsize=13,
        )
    return fig, axes


def plot_necessity(summary: Mapping[str, Any] | None):
    import matplotlib.pyplot as plt

    with publication_style(FigureTheme(dpi=600)):
        fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.7), constrained_layout=True)
        if summary is None:
            for ax in axes:
                ax.axis("off")
                ax.text(0.5, 0.5, "Not estimable: no reliable matched specialists",
                        ha="center", va="center")
            return fig, axes
        _errorbar_cells(
            axes[0],
            summary,
            "N_fraction",
            title="Direction-aligned event effect removed",
            ylabel="Necessity fraction",
        )
        _errorbar_cells(
            axes[1],
            summary,
            "N_gross_fraction",
            title="Gross event-response change",
            ylabel="Gross necessity / event effect",
        )
        for ax in axes:
            ax.axhline(1.0, color="#666666", linestyle=":", linewidth=0.9)
        fig.suptitle(
            "Donor-wise necessity: ablate the head in clean and intervened runs",
            fontsize=13,
        )
    return fig, axes


def plot_J_clean_ablation(clean: Mapping[str, Any], gate: Mapping[str, Any]):
    import matplotlib.pyplot as plt

    J = np.asarray(clean["J"], dtype=np.float64)
    movement = np.asarray(clean["prediction_movement"], dtype=np.float64)
    low = np.asarray(clean["prediction_movement_low"], dtype=np.float64)
    high = np.asarray(clean["prediction_movement_high"], dtype=np.float64)
    layers = np.asarray(clean["layers"], dtype=np.int64)
    shape = tuple(np.asarray(gate["masks"]["semantic_specialist"]).shape)
    codes, _ = _head_category(gate, shape)
    cmap = plt.get_cmap("viridis")
    normalizer = plt.Normalize(0, max(1, shape[0] - 1))

    with publication_style(FigureTheme(dpi=600)):
        fig, ax = plt.subplots(figsize=(7.2, 5.7), constrained_layout=True)
        ax.errorbar(
            J,
            movement,
            yerr=np.maximum(0.0, np.vstack((movement - low, high - movement))),
            fmt="none",
            ecolor="#777777",
            alpha=0.18,
            linewidth=0.45,
            zorder=1,
        )
        ax.scatter(
            J,
            movement,
            c=cmap(normalizer(layers)),
            s=28,
            edgecolors=np.where((codes.reshape(-1) < 2)[:, None],
                                np.asarray([0.05, 0.05, 0.05, 1.0]),
                                np.asarray([1.0, 1.0, 1.0, 0.35])),
            linewidths=np.where(codes.reshape(-1) < 2, 1.0, 0.3),
            alpha=0.92,
            zorder=2,
        )
        colorbar = fig.colorbar(
            plt.cm.ScalarMappable(norm=normalizer, cmap=cmap), ax=ax, pad=0.02
        )
        colorbar.set_label("Layer")
        ax.set_xlabel(r"Joint sensitivity $J$")
        ax.set_ylabel(r"Clean head-ablation movement $||z-z_{-h}||_2$")
        ax.set_title("Joint sensitivity predicts clean-input head importance")
        ax.grid(alpha=0.18, linewidth=0.5)
        ax.text(
            0.02,
            0.98,
            (
                f"Spearman $\\rho$={clean['spearman_rho']:.2f} "
                f"[{clean['spearman_low']:.2f}, {clean['spearman_high']:.2f}]\n"
                f"layer-adjusted $\\beta$={clean['layer_adjusted_standardized_beta']:.2f} "
                f"[{clean['layer_adjusted_low']:.2f}, {clean['layer_adjusted_high']:.2f}]\n"
                f"heads={clean['head_count']}, molecules={clean['graph_count']}"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
        )
    return fig, ax


def render_focused_figure_suite(
    scores: Mapping[str, Any],
    gate: Mapping[str, Any],
    core: Mapping[str, Any],
    *,
    output_dir: str | Path,
    common_metadata: Mapping[str, Any],
) -> dict[str, list[str]]:
    """Render the four frozen figures and their JSON provenance sidecars."""

    builder = FigureBuilder(
        Path(output_dir),
        theme=FigureTheme(dpi=600, formats=("pdf", "png")),
        common_metadata=dict(common_metadata),
    )
    outputs: dict[str, list[str]] = {}

    figure, axes, labels = plot_specialist_gate(scores, gate)
    paths = builder.save(
        "01_specialist_gate_map",
        figure,
        axes,
        metadata={
            "figure_role": "frozen specialist selection",
            "gate_status": gate["status"],
            "category_table": labels,
            "bootstrap_support": gate["support"],
            "molecule_diagnostic_fractions": {
                "semantic": gate["molecule_diagnostic"]["semantic_fraction"],
                "structural": gate["molecule_diagnostic"]["structural_fraction"],
                "central": gate["molecule_diagnostic"]["central_fraction"],
            },
            "specialist_J_matching": gate["specialist_J_matching"],
            "semantic_null_J_matching": gate["semantic_null_J_matching"],
            "structural_null_J_matching": gate["structural_null_J_matching"],
        },
    )
    outputs["specialist_gate"] = [str(path) for path in paths]

    figure, axes = plot_restoration_injection(core["patch"])
    paths = builder.save(
        "02_restoration_injection",
        figure,
        axes,
        metadata={
            "figure_role": "gross and direction-aligned causal patching",
            "patch_summary": core["patch"],
            "control_audit": core["control_audit"],
        },
    )
    outputs["restoration_injection"] = [str(path) for path in paths]

    figure, axes = plot_necessity(core["necessity"])
    paths = builder.save(
        "03_donor_wise_necessity",
        figure,
        axes,
        metadata={
            "figure_role": "donor-wise necessity fractions",
            "necessity_summary": core["necessity"],
        },
    )
    outputs["necessity"] = [str(path) for path in paths]

    figure, ax = plot_J_clean_ablation(core["clean_ablation"], gate)
    paths = builder.save(
        "04_J_vs_clean_ablation",
        figure,
        ax,
        metadata={
            "figure_role": "J versus clean single-head ablation",
            "clean_ablation_summary": core["clean_ablation"],
        },
    )
    outputs["J_clean_ablation"] = [str(path) for path in paths]
    return outputs


__all__ = [
    "plot_J_clean_ablation",
    "plot_necessity",
    "plot_restoration_injection",
    "plot_specialist_gate",
    "render_focused_figure_suite",
]
