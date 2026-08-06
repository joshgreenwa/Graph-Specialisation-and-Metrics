"""Second-pass, paper-composed mockups for the synthetic experiment section.

This script keeps grey as the visual base, reserves vermillion/blue for
semantic/structural identity, and composes the seven existing plot contents as
five manuscript figures.  All data are deterministic synthetic examples.
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


ROOT = Path(__file__).resolve().parents[1]
PNG_DIR = ROOT / "output" / "figure_redesign" / "paper"
PDF_WORK_DIR = ROOT / "tmp" / "figure_redesign" / "paper_pdfs"
FINAL_PDF = (
    ROOT
    / "output"
    / "pdf"
    / "figure_redesign"
    / "synthetic_figure_paper_suite.pdf"
)
LAYOUT_PREVIEW = ROOT / "output" / "figure_redesign" / "paper_section_layout.png"


TEXT_WIDTH = 6.85
SEMANTIC = base.SEMANTIC
STRUCTURAL = base.STRUCTURAL
GENERALIST = "#50555A"
INERT = "#A2A7AC"
NEUTRAL_BAND = "#F0F1F2"

# A quiet diverging scale for signed effects.  It reads as cool/warm grey rather
# than introducing a third categorical colour system.
EFFECT_CMAP = LinearSegmentedColormap.from_list(
    "paper_effect",
    ("#71808D", "#F6F4EF", "#6E5D55"),
)


def save_paper_figure(fig: mpl.figure.Figure, stem: str) -> tuple[Path, Path]:
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    PDF_WORK_DIR.mkdir(parents=True, exist_ok=True)
    png = PNG_DIR / f"{stem}.png"
    pdf = PDF_WORK_DIR / f"{stem}.pdf"
    metadata = {
        "Creator": "Graph Specialisation and Metrics",
        "Title": stem,
        "Subject": "Synthetic-data paper-layout figure mockup",
    }
    fig.savefig(png, dpi=base.PNG_DPI, facecolor=base.PAPER, bbox_inches=None)
    fig.savefig(
        pdf,
        dpi=base.PDF_RASTER_DPI,
        facecolor=base.PAPER,
        bbox_inches=None,
        metadata=metadata,
    )
    plt.close(fig)
    return png, pdf


def _compact_head_handles(*, selections: bool = False) -> tuple[list[Line2D], list[str]]:
    handles = base._layer_seed_handles()
    labels = ["L1", "L2", "L3", "seed 0", "seed 1", "seed 2"]
    if selections:
        handles.extend(base._selection_handles())
        labels.extend(["semantic selection", "structural selection"])
    return handles, labels


def _region_label(ax: mpl.axes.Axes, x: float, label: str, colour: str) -> None:
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


def paper_figure_1(points: Sequence[base.HeadPoint]) -> tuple[Path, Path]:
    """Raw scores and J/D plane as one full-width figure with one legend."""

    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 3.05))

    # (a) Raw task scores.
    ax = axes[0]
    structural = np.array([point.structural_score for point in points])
    semantic = np.array([point.semantic_score for point in points])
    base._scatter_heads(ax, points, structural, semantic, selected_rings=True)
    lo = 10.0 ** math.floor(math.log10(min(structural.min(), semantic.min())))
    hi = 10.0 ** math.ceil(math.log10(max(structural.max(), semantic.max())))
    ax.plot(
        [lo, hi],
        [lo, hi],
        color=base.REFERENCE,
        linewidth=0.82,
        linestyle=(0, (3.0, 2.2)),
        zorder=1,
    )
    ax.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi))
    ax.set_xlabel(r"Structural score $S_{\mathrm{str}}$")
    ax.set_ylabel(r"Semantic score $S_{\mathrm{sem}}$")
    ax.set_box_aspect(1.0)
    base.panel_label(ax, "(a)", "Raw task scores")
    base.style_axes(ax, xgrid=True, ygrid=True)

    # (b) Joint sensitivity and task selectivity.
    ax = axes[1]
    selectivity = np.array([point.selectivity for point in points])
    sensitivity = np.array([point.joint_sensitivity for point in points])
    ax.axvspan(-0.20, 0.20, color=NEUTRAL_BAND, linewidth=0, zorder=0)
    ax.axvline(-0.20, color=base.GRID, linewidth=0.55, zorder=1)
    ax.axvline(0.20, color=base.GRID, linewidth=0.55, zorder=1)
    base._scatter_heads(ax, points, selectivity, sensitivity)
    ax.axvline(0.0, color=base.REFERENCE, linewidth=0.68, zorder=1)
    ax.axhline(
        0.5,
        color=base.REFERENCE,
        linewidth=0.80,
        linestyle=(0, (3.0, 2.2)),
        zorder=1,
    )
    _region_label(ax, 0.18, "structural", STRUCTURAL)
    _region_label(ax, 0.50, "generalist", base.MUTED)
    _region_label(ax, 0.82, "semantic", SEMANTIC)
    ax.annotate(
        r"$J=0.5$",
        xy=(-0.98, 0.5),
        xytext=(3, 3),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=7.0,
        color=base.MUTED,
        bbox={"facecolor": base.PAPER, "edgecolor": "none", "alpha": 0.78, "pad": 0.35},
    )
    ax.set_xlim(-1.02, 1.02)
    ax.set_yscale("log")
    ax.set_xlabel(r"Head selectivity $D_{\mathrm{rel}}$")
    ax.set_ylabel(r"Joint sensitivity $J$")
    ax.set_box_aspect(1.0)
    base.panel_label(ax, "(b)", "Sensitivity and selectivity")
    base.style_axes(ax, xgrid=False, ygrid=True)

    handles, labels = _compact_head_handles(selections=True)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.018),
        ncol=8,
        columnspacing=0.58,
        handletextpad=0.28,
        borderaxespad=0,
        fontsize=7.0,
    )
    fig.subplots_adjust(left=0.085, right=0.992, bottom=0.205, top=0.88, wspace=0.31)
    return save_paper_figure(fig, "paper_fig1_head_metrics")


def _rho_text(ax: mpl.axes.Axes, text: str) -> None:
    ax.text(
        0.04,
        0.93,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.0,
        color=base.MUTED,
        linespacing=1.18,
    )


def paper_figure_2(
    points: Sequence[base.HeadPoint],
    seed: int = 771,
) -> tuple[Path, Path]:
    """Causal validation triptych with compact statistical annotation."""

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

    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 2.52))
    specs = (
        (
            sensitivity,
            impact,
            r"Joint sensitivity $J$",
            "Cross-task ablation impact",
            "Sensitivity vs impact",
            False,
        ),
        (
            selectivity,
            ablation,
            r"Score selectivity $D_{\mathrm{rel}}$",
            "Ablation role contrast",
            "Selectivity vs ablation",
            True,
        ),
        (
            selectivity,
            rescue,
            r"Score selectivity $D_{\mathrm{rel}}$",
            "Rescue role contrast",
            "Selectivity vs rescue",
            True,
        ),
    )

    for index, (ax, spec) in enumerate(zip(axes, specs)):
        x, y, xlabel, ylabel, title, dim_unreliable = spec
        base._scatter_heads(
            ax,
            points,
            x,
            y,
            dim_unreliable=dim_unreliable,
        )
        base.panel_label(ax, f"({chr(97 + index)})", title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if index == 0:
            ax.set_xscale("log")
            ax.set_yscale("log")
            _rho_text(ax, rf"$\rho={base._spearman(x, y):.2f}$")
        else:
            ax.axhline(0.0, color=base.REFERENCE, linewidth=0.64, zorder=1)
            ax.axvline(0.0, color=base.REFERENCE, linewidth=0.64, zorder=1)
            ax.set_xlim(-1.03, 1.03)
            if index == 1:
                ax.set_ylim(-1.03, 1.03)
            rho_rel = base._spearman(x[reliable], y[reliable])
            _rho_text(
                ax,
                rf"$\rho_{{\rm rel}}={rho_rel:.2f}$",
            )
        base.style_axes(ax, xgrid=False, ygrid=True)

    handles, labels = _compact_head_handles()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=6,
        columnspacing=0.82,
        handletextpad=0.30,
        borderaxespad=0,
        fontsize=7.0,
    )
    fig.subplots_adjust(left=0.073, right=0.992, bottom=0.285, top=0.82, wspace=0.44)
    return save_paper_figure(fig, "paper_fig2_causal_validation")


FAMILY_STYLES = {
    "semantic": (SEMANTIC, "o", "-", "Semantic"),
    "structural": (STRUCTURAL, "s", (0, (4.0, 1.8)), "Structural"),
    "generalist": (GENERALIST, "D", (0, (2.3, 1.4)), "Generalist"),
    "inert": (INERT, "^", (0, (1.0, 1.6)), "Inert"),
}


def paper_figure_3() -> tuple[Path, Path]:
    """Joint ablation grid: mean and observed seed range."""

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(TEXT_WIDTH, 4.25),
        sharex=True,
        sharey="row",
    )
    tasks = ("semantic", "structural")
    metrics = (
        ("functional", "Functional logit impact"),
        ("accuracy", "Accuracy drop (pp)"),
    )
    for col, task in enumerate(tasks):
        for row, (metric, ylabel) in enumerate(metrics):
            ax = axes[row, col]
            for family, (colour, marker, linestyle, label) in FAMILY_STYLES.items():
                curves = base._ablation_curves(
                    family,
                    task,
                    metric,
                    seed=711 + row * 31 + col * 17,
                )
                x = np.arange(curves.shape[1])
                mean = curves.mean(axis=0)
                lower = curves.min(axis=0)
                upper = curves.max(axis=0)
                ax.fill_between(
                    x,
                    lower,
                    upper,
                    color=colour,
                    alpha=0.075,
                    linewidth=0,
                )
                ax.plot(
                    x,
                    mean,
                    color=colour,
                    marker=marker,
                    linestyle=linestyle,
                    linewidth=1.65,
                    markersize=3.4,
                    markeredgecolor=base.PAPER,
                    markeredgewidth=0.35,
                    label=label,
                    zorder=3,
                )
            ax.axhline(0.0, color=base.REFERENCE, linewidth=0.66, zorder=1)
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == 0:
                ax.set_title(f"{task.capitalize()} task")
            base.panel_label(ax, f"({chr(97 + row * 2 + col)})")
            ax.set_xticks(np.arange(4))
            base.style_axes(ax, xgrid=False, ygrid=True)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=4,
        columnspacing=1.10,
        handlelength=2.45,
        handletextpad=0.42,
        fontsize=7.0,
    )
    fig.supxlabel("Jointly ablated heads", x=0.54, y=0.075, fontsize=8.8)
    fig.subplots_adjust(left=0.105, right=0.982, bottom=0.17, top=0.90, hspace=0.25, wspace=0.22)
    return save_paper_figure(fig, "paper_fig3_joint_ablation")


def _draw_matrix_panel(
    fig: mpl.figure.Figure,
    ax: mpl.axes.Axes,
    colorbar_ax: mpl.axes.Axes,
    matrix: np.ndarray,
    *,
    panel: str,
    title: str,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    xlabel: str,
    ylabel: str,
    colorbar_label: str,
    ticks: Sequence[float],
) -> None:
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
                fontsize=8.4,
                fontweight="semibold",
                color=base.INK if luminance > 0.57 else base.PAPER,
            )
    ax.set_xlim(-0.5, matrix.shape[1] - 0.5)
    ax.set_ylim(matrix.shape[0] - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(len(col_labels)), col_labels)
    ax.set_yticks(np.arange(len(row_labels)), row_labels)
    ax.tick_params(axis="both", length=0, pad=4, labelsize=7.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    base.panel_label(ax, panel, title)
    for spine in ax.spines.values():
        spine.set_visible(False)

    scalar = mpl.cm.ScalarMappable(norm=norm, cmap=EFFECT_CMAP)
    colorbar = fig.colorbar(
        scalar,
        cax=colorbar_ax,
        orientation="horizontal",
        ticks=ticks,
    )
    colorbar.set_label(colorbar_label, labelpad=2.0, fontsize=7.0)
    colorbar.outline.set_linewidth(0.5)
    colorbar.outline.set_edgecolor(base.AXIS)
    colorbar.ax.tick_params(labelsize=7.0, length=2.0, pad=1.5)


def paper_figure_4() -> tuple[Path, Path]:
    """Necessity and rescue matrices as one aligned paper figure."""

    fig = plt.figure(figsize=(TEXT_WIDTH, 2.75))
    axes = (
        fig.add_axes([0.17, 0.30, 0.25, 0.59]),
        fig.add_axes([0.64, 0.30, 0.25, 0.59]),
    )
    colorbar_axes = (
        fig.add_axes([0.17, 0.095, 0.25, 0.036]),
        fig.add_axes([0.64, 0.095, 0.25, 0.036]),
    )
    _draw_matrix_panel(
        fig,
        axes[0],
        colorbar_axes[0],
        np.array([[0.004, 0.052], [-0.001, 0.011]], dtype=float),
        panel="(a)",
        title="Necessity",
        row_labels=("Semantic", "Structural"),
        col_labels=("Semantic", "Structural"),
        xlabel="Evaluation task",
        ylabel="Ablated family",
        colorbar_label=r"Change in cross-entropy ($\Delta$CE)",
        ticks=(-0.05, 0.00, 0.05),
    )
    _draw_matrix_panel(
        fig,
        axes[1],
        colorbar_axes[1],
        np.array([[0.154, 0.066], [-0.008, 0.128]], dtype=float),
        panel="(b)",
        title="Rescue",
        row_labels=("Semantic", "Structural"),
        col_labels=("Semantic", "Structural"),
        xlabel="Corruption",
        ylabel="Patched family",
        colorbar_label="Mediated-effect fraction",
        ticks=(-0.15, 0.00, 0.15),
    )
    return save_paper_figure(fig, "paper_fig4_dissociation_matrices")


def paper_figure_5() -> tuple[Path, Path]:
    """Reuse the already-strong attention redesign as the final paper figure."""

    source_png, source_pdf = base.figure_7()
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    PDF_WORK_DIR.mkdir(parents=True, exist_ok=True)
    png = PNG_DIR / "paper_fig5_attention.png"
    pdf = PDF_WORK_DIR / "paper_fig5_attention.pdf"
    shutil.copyfile(source_png, png)
    shutil.copyfile(source_pdf, pdf)
    return png, pdf


def combine_pdfs(pdfs: Sequence[Path], output: Path) -> None:
    from pypdf import PdfWriter

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = PdfWriter()
    for pdf in pdfs:
        writer.append(str(pdf))
    writer.add_metadata(
        {
            "/Title": "Synthetic experiment paper figure suite",
            "/Author": "Graph Specialisation and Metrics",
            "/Subject": "Five paper-composed synthetic-data figure mockups",
        }
    )
    with output.open("wb") as stream:
        writer.write(stream)


def make_layout_preview(pngs: Sequence[Path], output: Path) -> None:
    from PIL import Image

    full_width = 1600
    gap = 34
    rows: list[Image.Image] = []
    for path in pngs:
        image = Image.open(path).convert("RGB")
        rows.append(
            image.resize(
                (full_width, round(image.height * full_width / image.width)),
                Image.Resampling.LANCZOS,
            )
        )
    canvas = Image.new(
        "RGB",
        (full_width, sum(row.height for row in rows) + gap * (len(rows) - 1)),
        "#E8EBEE",
    )
    top = 0
    for row in rows:
        canvas.paste(row, (0, top))
        top += row.height + gap
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, dpi=(180, 180), quality=95)


def main() -> None:
    base.configure_matplotlib()
    points = base.generate_head_points()
    outputs = [
        paper_figure_1(points),
        paper_figure_2(points),
        paper_figure_3(),
        paper_figure_4(),
        paper_figure_5(),
    ]
    pngs = [png for png, _pdf in outputs]
    pdfs = [pdf for _png, pdf in outputs]
    combine_pdfs(pdfs, FINAL_PDF)
    make_layout_preview(pngs, LAYOUT_PREVIEW)
    print(f"Wrote {len(pngs)} paper-composed PNGs to {PNG_DIR}")
    print(f"Wrote paper suite PDF to {FINAL_PDF}")
    print(f"Wrote paper layout preview to {LAYOUT_PREVIEW}")


if __name__ == "__main__":
    main()
