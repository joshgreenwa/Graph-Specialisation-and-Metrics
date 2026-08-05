"""Matched component figures for side-by-side manuscript placement.

The synthetic content mirrors the notebook's nine paper assets after splitting:
two head-metric panels, three causal panels, one ablation grid, two matrices, and
one attention grid.  All data are deterministic mock values.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence
import math
import shutil

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np

import mock_figure_suite as base
import paper_figure_suite as paper


ROOT = Path(__file__).resolve().parents[1]
PNG_DIR = ROOT / "output" / "figure_redesign" / "side_by_side"
PDF_WORK_DIR = ROOT / "tmp" / "figure_redesign" / "side_by_side_pdfs"
FINAL_PDF = (
    ROOT
    / "output"
    / "pdf"
    / "figure_redesign"
    / "synthetic_figure_side_by_side_suite.pdf"
)
LAYOUT_PREVIEW = (
    ROOT / "output" / "figure_redesign" / "side_by_side_paper_layout.png"
)


TEXT_WIDTH = 6.85
PAIR_GAP = 0.22
TRIPTYCH_GAP = 0.16
HALF_WIDTH = (TEXT_WIDTH - PAIR_GAP) / 2.0
THIRD_WIDTH = (TEXT_WIDTH - 2.0 * TRIPTYCH_GAP) / 3.0

PAIR_SIZE = (HALF_WIDTH, 3.12)
CAUSAL_SIZE = (THIRD_WIDTH, 2.36)
MATRIX_SIZE = (HALF_WIDTH, 2.68)

SEMANTIC = base.SEMANTIC
STRUCTURAL = base.STRUCTURAL
GENERALIST = "#50555A"
INERT = "#A2A7AC"
NEUTRAL_BAND = "#F0F1F2"

EFFECT_CMAP = LinearSegmentedColormap.from_list(
    "effect_blue_vermillion",
    ("#3B75AF", "#F7F7F5", "#C84E3A"),
)


def save_component(fig: mpl.figure.Figure, stem: str) -> tuple[Path, Path]:
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    PDF_WORK_DIR.mkdir(parents=True, exist_ok=True)
    png = PNG_DIR / f"{stem}.png"
    pdf = PDF_WORK_DIR / f"{stem}.pdf"
    metadata = {
        "Creator": "Graph Specialisation and Metrics",
        "Title": stem,
        "Subject": "Synthetic-data side-by-side figure mockup",
    }
    fig.savefig(png, dpi=300, facecolor=base.PAPER, bbox_inches=None)
    fig.savefig(pdf, dpi=300, facecolor=base.PAPER, bbox_inches=None, metadata=metadata)
    plt.close(fig)
    return png, pdf


def _head_handles() -> tuple[list[Line2D], list[str]]:
    handles = []
    labels = []
    for layer, colour in enumerate(base.LAYER_COLOURS):
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=colour,
                markeredgecolor=base.AXIS,
                markeredgewidth=0.35,
                markersize=5.1,
            )
        )
        labels.append(f"L{layer + 1}")
    for seed, marker in enumerate(base.SEED_MARKERS):
        handles.append(
            Line2D(
                [0],
                [0],
                marker=marker,
                linestyle="none",
                markerfacecolor=base.PAPER,
                markeredgecolor=base.MUTED,
                markeredgewidth=0.8,
                markersize=5.0,
            )
        )
        labels.append(f"seed {seed}")
    return handles, labels


def _selection_handles() -> tuple[list[Line2D], list[str]]:
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=base.PAPER,
            markeredgecolor=SEMANTIC,
            markeredgewidth=1.5,
            markersize=5.8,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=base.PAPER,
            markeredgecolor=STRUCTURAL,
            markeredgewidth=1.5,
            markersize=5.8,
        ),
    ]
    return handles, ["semantic selection", "structural selection"]


def draw_heads(
    ax: mpl.axes.Axes,
    points: Sequence[base.HeadPoint],
    x_values: Sequence[float],
    y_values: Sequence[float],
    *,
    selected_rings: bool = False,
    dim_unreliable: bool = False,
    marker_size: float = 27.0,
) -> None:
    for point, x, y in zip(points, x_values, y_values):
        reliable = point.joint_sensitivity >= 0.5
        alpha = 0.90 if (reliable or not dim_unreliable) else 0.16
        marker = base.SEED_MARKERS[point.seed]
        ax.scatter(
            x,
            y,
            s=marker_size,
            marker=marker,
            facecolor=base.LAYER_COLOURS[point.layer],
            edgecolor=base.PAPER,
            linewidth=0.45,
            alpha=alpha,
            zorder=3,
        )
        selected = (
            selected_rings
            and point.family in {"semantic", "structural"}
            and point.head in {0, 2}
        )
        if selected:
            ring = SEMANTIC if point.family == "semantic" else STRUCTURAL
            ax.scatter(
                x,
                y,
                s=marker_size + 23,
                marker=marker,
                facecolor="none",
                edgecolor=base.PAPER,
                linewidth=2.5,
                zorder=4,
            )
            ax.scatter(
                x,
                y,
                s=marker_size + 23,
                marker=marker,
                facecolor="none",
                edgecolor=ring,
                linewidth=1.35,
                zorder=5,
            )


def _shared_head_legend(fig: mpl.figure.Figure, *, y: float) -> None:
    handles, labels = _head_handles()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        ncol=6,
        columnspacing=0.48,
        handletextpad=0.24,
        borderaxespad=0,
        fontsize=7.0,
    )


def raw_score_figure(points: Sequence[base.HeadPoint]) -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=PAIR_SIZE)
    structural = np.array([point.structural_score for point in points])
    semantic = np.array([point.semantic_score for point in points])
    draw_heads(ax, points, structural, semantic, selected_rings=True)
    lo = 10.0 ** math.floor(math.log10(min(structural.min(), semantic.min())))
    hi = 10.0 ** math.ceil(math.log10(max(structural.max(), semantic.max())))
    ax.plot(
        [lo, hi],
        [lo, hi],
        color=base.REFERENCE,
        linewidth=0.78,
        linestyle=(0, (3.0, 2.2)),
        zorder=1,
    )
    ax.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi))
    ax.set_xlabel(r"Structural score $S_{\mathrm{str}}$")
    ax.set_ylabel(r"Semantic score $S_{\mathrm{sem}}$")
    ax.set_box_aspect(1.0)
    base.panel_label(ax, "(a)", "Raw task scores")
    base.style_axes(ax, xgrid=True, ygrid=True)
    _shared_head_legend(fig, y=0.085)
    selection_handles, selection_labels = _selection_handles()
    fig.legend(
        selection_handles,
        selection_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=2,
        columnspacing=0.80,
        handletextpad=0.30,
        borderaxespad=0,
        fontsize=7.0,
    )
    fig.subplots_adjust(left=0.19, right=0.975, bottom=0.29, top=0.88)
    return save_component(fig, "side_fig1a_raw_scores")


def joint_selectivity_figure(points: Sequence[base.HeadPoint]) -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=PAIR_SIZE)
    selectivity = np.array([point.selectivity for point in points])
    sensitivity = np.array([point.joint_sensitivity for point in points])
    ax.axvspan(-0.20, 0.20, color=NEUTRAL_BAND, linewidth=0, zorder=0)
    ax.axvline(-0.20, color=base.GRID, linewidth=0.52, zorder=1)
    ax.axvline(0.20, color=base.GRID, linewidth=0.52, zorder=1)
    draw_heads(ax, points, selectivity, sensitivity)
    ax.axvline(0.0, color=base.REFERENCE, linewidth=0.66, zorder=1)
    ax.axhline(
        0.5,
        color=base.REFERENCE,
        linewidth=0.78,
        linestyle=(0, (3.0, 2.2)),
        zorder=1,
    )
    for x, label, colour in (
        (0.18, "structural", STRUCTURAL),
        (0.50, "generalist", base.MUTED),
        (0.82, "semantic", SEMANTIC),
    ):
        ax.text(
            x,
            0.965,
            label,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=7.0,
            color=colour,
            bbox={"facecolor": base.PAPER, "edgecolor": "none", "alpha": 0.78, "pad": 0.4},
            zorder=6,
        )
    ax.annotate(
        r"$J=0.5$",
        xy=(-0.98, 0.5),
        xytext=(3, 3),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=7.0,
        color=base.MUTED,
    )
    ax.set_xlim(-1.02, 1.02)
    ax.set_yscale("log")
    ax.set_xlabel(r"Head selectivity $D_{\mathrm{rel}}$")
    ax.set_ylabel(r"Joint sensitivity $J$")
    ax.set_box_aspect(1.0)
    base.panel_label(ax, "(b)", "Sensitivity and selectivity")
    base.style_axes(ax, xgrid=False, ygrid=True)
    _shared_head_legend(fig, y=0.085)
    fig.subplots_adjust(left=0.19, right=0.975, bottom=0.29, top=0.88)
    return save_component(fig, "side_fig1b_joint_selectivity")


def _rho_note(ax: mpl.axes.Axes, text: str) -> None:
    ax.text(
        0.04,
        0.93,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.0,
        color=base.MUTED,
    )


def causal_figures(
    points: Sequence[base.HeadPoint],
    seed: int = 771,
) -> list[tuple[Path, Path]]:
    rng = np.random.default_rng(seed)
    sensitivity = np.array([point.joint_sensitivity for point in points])
    selectivity = np.array([point.selectivity for point in points])
    reliable = sensitivity >= 0.5
    impact = np.maximum(
        0.012,
        0.20 * sensitivity**0.86 * np.exp(rng.normal(0.0, 0.40, sensitivity.size)),
    )
    ablation = np.clip(
        0.84 * selectivity + rng.normal(0.0, 0.18, selectivity.size),
        -1.0,
        1.0,
    )
    rescue = 0.15 * selectivity + rng.normal(0.0, 0.052, selectivity.size)
    specs = (
        (
            sensitivity,
            impact,
            r"Joint sensitivity $J$",
            "Ablation impact",
            "(a) Sensitivity vs impact",
            "causal_a_sensitivity_impact",
            False,
        ),
        (
            selectivity,
            ablation,
            r"Selectivity $D_{\mathrm{rel}}$",
            "Ablation role",
            "(b) Selectivity vs ablation",
            "causal_b_selectivity_ablation",
            True,
        ),
        (
            selectivity,
            rescue,
            r"Selectivity $D_{\mathrm{rel}}$",
            "Rescue role",
            "(c) Selectivity vs rescue",
            "causal_c_selectivity_rescue",
            True,
        ),
    )
    outputs = []
    for index, spec in enumerate(specs):
        x, y, xlabel, ylabel, title, stem, dim_unreliable = spec
        fig, ax = plt.subplots(figsize=CAUSAL_SIZE)
        draw_heads(
            ax,
            points,
            x,
            y,
            dim_unreliable=dim_unreliable,
            marker_size=21.0,
        )
        ax.set_title(title, loc="left", fontsize=8.2, fontweight="semibold", pad=4.0)
        ax.set_xlabel(xlabel, fontsize=8.0)
        ax.set_ylabel(ylabel, fontsize=8.0)
        if index == 0:
            ax.set_xscale("log")
            ax.set_yscale("log")
            _rho_note(ax, rf"$\rho={base._spearman(x, y):.2f}$")
        else:
            ax.axhline(0.0, color=base.REFERENCE, linewidth=0.62, zorder=1)
            ax.axvline(0.0, color=base.REFERENCE, linewidth=0.62, zorder=1)
            ax.set_xlim(-1.03, 1.03)
            if index == 1:
                ax.set_ylim(-1.03, 1.03)
            _rho_note(ax, rf"$\rho_{{\rm rel}}={base._spearman(x[reliable], y[reliable]):.2f}$")
        base.style_axes(ax, xgrid=False, ygrid=True)
        fig.subplots_adjust(left=0.235, right=0.975, bottom=0.22, top=0.84)
        outputs.append(save_component(fig, stem))
    return outputs


def _draw_matrix(
    matrix: np.ndarray,
    *,
    title: str,
    stem: str,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    xlabel: str,
    ylabel: str,
    colorbar_label: str,
    ticks: Sequence[float],
) -> tuple[Path, Path]:
    fig = plt.figure(figsize=MATRIX_SIZE)
    ax = fig.add_axes([0.30, 0.31, 0.57, 0.57])
    colorbar_ax = fig.add_axes([0.30, 0.095, 0.57, 0.036])
    vmax = max(abs(float(np.nanmin(matrix))), abs(float(np.nanmax(matrix))), 1.0e-6)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            colour = EFFECT_CMAP(norm(float(matrix[row, col])))
            ax.add_patch(
                Rectangle(
                    (col - 0.5, row - 0.5),
                    1.0,
                    1.0,
                    facecolor=colour,
                    edgecolor=base.PAPER,
                    linewidth=1.15,
                )
            )
            red, green, blue, _ = colour
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            ax.text(
                col,
                row,
                f"{float(matrix[row, col]):+.3f}",
                ha="center",
                va="center",
                fontsize=8.3,
                fontweight="semibold",
                color=base.INK if luminance > 0.57 else base.PAPER,
            )
    ax.set_xlim(-0.5, matrix.shape[1] - 0.5)
    ax.set_ylim(matrix.shape[0] - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(len(col_labels)), col_labels)
    ax.set_yticks(np.arange(len(row_labels)), row_labels)
    ax.tick_params(axis="both", length=0, pad=4, labelsize=7.0)
    ax.set_xlabel(xlabel, fontsize=8.2)
    ax.set_ylabel(ylabel, fontsize=8.2)
    ax.set_title(title, loc="left", fontsize=8.6, fontweight="semibold", pad=4.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    scalar = mpl.cm.ScalarMappable(norm=norm, cmap=EFFECT_CMAP)
    colorbar = fig.colorbar(scalar, cax=colorbar_ax, orientation="horizontal", ticks=ticks)
    colorbar.set_label(colorbar_label, labelpad=1.8, fontsize=7.0)
    colorbar.outline.set_linewidth(0.5)
    colorbar.outline.set_edgecolor(base.AXIS)
    colorbar.ax.tick_params(labelsize=7.0, length=2.0, pad=1.3)
    return save_component(fig, stem)


def matrix_figures() -> list[tuple[Path, Path]]:
    necessity = _draw_matrix(
        np.array([[0.004, 0.052], [-0.001, 0.011]], dtype=float),
        title="(a) Necessity",
        stem="side_fig4a_necessity",
        row_labels=("Semantic", "Structural"),
        col_labels=("Semantic", "Structural"),
        xlabel="Evaluation task",
        ylabel="Ablated family",
        colorbar_label=r"Change in cross-entropy ($\Delta$CE)",
        ticks=(-0.05, 0.00, 0.05),
    )
    rescue = _draw_matrix(
        np.array([[0.154, 0.066], [-0.008, 0.128]], dtype=float),
        title="(b) Rescue",
        stem="side_fig4b_rescue",
        row_labels=("Semantic", "Structural"),
        col_labels=("Semantic", "Structural"),
        xlabel="Corruption",
        ylabel="Patched family",
        colorbar_label="Mediated-effect fraction",
        ticks=(-0.15, 0.00, 0.15),
    )
    return [necessity, rescue]


def _copy_full_width_figures() -> tuple[tuple[Path, Path], tuple[Path, Path]]:
    ablation_png, ablation_pdf = paper.paper_figure_3()
    attention_png, attention_pdf = base.figure_7()
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    PDF_WORK_DIR.mkdir(parents=True, exist_ok=True)
    copied = []
    for source_png, source_pdf, stem in (
        (ablation_png, ablation_pdf, "side_fig3_joint_ablation"),
        (attention_png, attention_pdf, "side_fig5_attention"),
    ):
        png = PNG_DIR / f"{stem}.png"
        pdf = PDF_WORK_DIR / f"{stem}.pdf"
        shutil.copyfile(source_png, png)
        shutil.copyfile(source_pdf, pdf)
        copied.append((png, pdf))
    return copied[0], copied[1]


def combine_pdfs(pdfs: Sequence[Path], output: Path) -> None:
    from pypdf import PdfWriter

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = PdfWriter()
    for pdf in pdfs:
        writer.append(str(pdf))
    writer.add_metadata(
        {
            "/Title": "Synthetic experiment side-by-side component suite",
            "/Author": "Graph Specialisation and Metrics",
            "/Subject": "Nine matched component figure mockups",
        }
    )
    with output.open("wb") as stream:
        writer.write(stream)


def make_layout_preview(
    raw: Path,
    joint: Path,
    causal: Sequence[Path],
    ablation: Path,
    matrices: Sequence[Path],
    attention: Path,
    output: Path,
) -> None:
    from PIL import Image

    full_width = 1650
    pair_gap = 54
    triple_gap = 38
    section_gap = 34
    half_width = (full_width - pair_gap) // 2
    third_width = (full_width - 2 * triple_gap) // 3

    def load_fit(path: Path, width: int) -> Image.Image:
        image = Image.open(path).convert("RGB")
        return image.resize(
            (width, round(image.height * width / image.width)),
            Image.Resampling.LANCZOS,
        )

    rows: list[Image.Image] = []
    pair = [load_fit(raw, half_width), load_fit(joint, half_width)]
    pair_row = Image.new("RGB", (full_width, max(image.height for image in pair)), base.PAPER)
    pair_row.paste(pair[0], (0, 0))
    pair_row.paste(pair[1], (half_width + pair_gap, 0))
    rows.append(pair_row)

    triple = [load_fit(path, third_width) for path in causal]
    triple_row = Image.new("RGB", (full_width, max(image.height for image in triple)), base.PAPER)
    for index, image in enumerate(triple):
        triple_row.paste(image, (index * (third_width + triple_gap), 0))
    rows.append(triple_row)

    rows.append(load_fit(ablation, full_width))

    pair = [load_fit(path, half_width) for path in matrices]
    matrix_row = Image.new("RGB", (full_width, max(image.height for image in pair)), base.PAPER)
    matrix_row.paste(pair[0], (0, 0))
    matrix_row.paste(pair[1], (half_width + pair_gap, 0))
    rows.append(matrix_row)

    rows.append(load_fit(attention, full_width))

    canvas = Image.new(
        "RGB",
        (full_width, sum(row.height for row in rows) + section_gap * (len(rows) - 1)),
        "#E8EBEE",
    )
    top = 0
    for row in rows:
        canvas.paste(row, (0, top))
        top += row.height + section_gap
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, dpi=(180, 180), quality=95)


def main() -> None:
    base.configure_matplotlib()
    points = base.generate_head_points()
    raw = raw_score_figure(points)
    joint = joint_selectivity_figure(points)
    causal = causal_figures(points)
    ablation, attention = _copy_full_width_figures()
    matrices = matrix_figures()

    ordered = [raw, joint, *causal, ablation, *matrices, attention]
    combine_pdfs([pdf for _png, pdf in ordered], FINAL_PDF)
    make_layout_preview(
        raw[0],
        joint[0],
        [output[0] for output in causal],
        ablation[0],
        [output[0] for output in matrices],
        attention[0],
        LAYOUT_PREVIEW,
    )
    print(f"Wrote {len(ordered)} component PNGs to {PNG_DIR}")
    print(f"Wrote component suite PDF to {FINAL_PDF}")
    print(f"Wrote side-by-side layout preview to {LAYOUT_PREVIEW}")


if __name__ == "__main__":
    main()
