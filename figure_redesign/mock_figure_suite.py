"""Publication-style mockups for the seven synthetic-experiment figures.

The data in this module are deterministic and synthetic.  The plotting code is
deliberately close to plain Matplotlib so that the selected style can later be
ported into the single-cell Colab notebook without adding new dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
import math

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PNG_DIR = ROOT / "output" / "figure_redesign" / "png"
PDF_WORK_DIR = ROOT / "tmp" / "figure_redesign" / "mock_pdfs"
FINAL_PDF = (
    ROOT
    / "output"
    / "pdf"
    / "figure_redesign"
    / "synthetic_figure_redesign_suite.pdf"
)
CONTACT_SHEET = ROOT / "output" / "figure_redesign" / "redesign_contact_sheet.png"
LAYOUT_PREVIEW = ROOT / "output" / "figure_redesign" / "redesign_section_layout.png"


# Semantic colours are stable across the complete suite.
SEMANTIC = "#D55E00"
STRUCTURAL = "#0072B2"
GENERALIST = "#7A5EA6"
INERT = "#7F858B"

# Layer identity uses a restrained categorical palette.  It remains distinct
# from semantic vermillion, structural blue, and continuous attention colour.
LAYER_COLOURS = ("#6B5B95", "#2A8C82", "#B88720")
SEED_MARKERS = ("o", "s", "^")

# Neutral construction colours.
INK = "#252A31"
MUTED = "#68717A"
AXIS = "#3D434B"
GRID = "#E7EAEE"
REFERENCE = "#8E969E"
PAPER = "#FFFFFF"
QUERY = "#20252B"
SOURCE = "#949CA4"

# Fixed canvases are chosen for final placement rather than notebook display.
TEXT_WIDTH = 6.85
GUTTER = 0.22
HALF_WIDTH = (TEXT_WIDTH - GUTTER) / 2.0
SINGLE_SIZE = (HALF_WIDTH, 3.05)
TRIPTYCH_SIZE = (TEXT_WIDTH, 2.60)
QUAD_SIZE = (TEXT_WIDTH, 4.75)
HEATMAP_SIZE = (HALF_WIDTH, 2.65)
ATTENTION_SIZE = (TEXT_WIDTH, 4.75)


@dataclass(frozen=True)
class HeadPoint:
    layer: int
    seed: int
    head: int
    structural_score: float
    semantic_score: float
    joint_sensitivity: float
    selectivity: float
    family: str


def configure_matplotlib() -> None:
    """Portable, Colab-safe publication defaults."""

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "font.weight": "regular",
            "mathtext.fontset": "dejavusans",
            "axes.titlesize": 9.0,
            "axes.titleweight": "semibold",
            "axes.titlepad": 5.0,
            "axes.labelsize": 8.8,
            "axes.labelcolor": INK,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.72,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 7.7,
            "ytick.labelsize": 7.7,
            "xtick.color": AXIS,
            "ytick.color": AXIS,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.minor.size": 1.8,
            "ytick.minor.size": 1.8,
            "legend.fontsize": 7.4,
            "legend.title_fontsize": 7.4,
            "legend.frameon": False,
            "text.color": INK,
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "savefig.facecolor": PAPER,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "savefig.bbox": None,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.unicode_minus": True,
        }
    )


def style_axes(
    ax: mpl.axes.Axes,
    *,
    xgrid: bool = False,
    ygrid: bool = True,
    log_minor_grid: bool = False,
) -> None:
    """Apply one restrained grid and spine treatment throughout."""

    if xgrid:
        ax.grid(axis="x", which="major", color=GRID, linewidth=0.42, alpha=0.74)
    if ygrid:
        ax.grid(axis="y", which="major", color=GRID, linewidth=0.42, alpha=0.74)
    if log_minor_grid:
        ax.grid(axis="both", which="minor", color=GRID, linewidth=0.34, alpha=0.35)
    ax.set_axisbelow(True)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.72)


def panel_label(ax: mpl.axes.Axes, label: str, title: str | None = None) -> None:
    ax.text(
        -0.02,
        1.055,
        label,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontweight="semibold",
        fontsize=8.7,
        clip_on=False,
    )
    if title:
        ax.set_title(title, loc="left")


def _layer_seed_handles() -> list[Line2D]:
    handles: list[Line2D] = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=colour,
            markeredgecolor=AXIS,
            markeredgewidth=0.35,
            markersize=5.4,
            label=f"Layer {layer + 1}",
        )
        for layer, colour in enumerate(LAYER_COLOURS)
    ]
    handles.extend(
        Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor=MUTED,
            markeredgewidth=0.8,
            markersize=5.2,
            label=f"Seed {seed}",
        )
        for seed, marker in enumerate(SEED_MARKERS)
    )
    return handles


def _selection_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor=SEMANTIC,
            markeredgewidth=1.5,
            markersize=5.8,
            label="Semantic-selected",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor=STRUCTURAL,
            markeredgewidth=1.5,
            markersize=5.8,
            label="Structural-selected",
        ),
    ]


def add_head_key(
    fig: mpl.figure.Figure,
    *,
    selection: bool = False,
    compact: bool = False,
    y: float = 0.02,
) -> None:
    handles = _layer_seed_handles()
    if selection:
        handles.extend(_selection_handles())
    labels = [handle.get_label() for handle in handles]
    if compact:
        labels = ["L1", "L2", "L3", "seed 0", "seed 1", "seed 2"]
        if selection:
            labels.extend(["Semantic-selected", "Structural-selected"])
    if selection and compact:
        fig.legend(
            handles=handles[:6],
            labels=labels[:6],
            loc="lower center",
            bbox_to_anchor=(0.5, y + 0.064),
            ncol=6,
            columnspacing=0.42,
            handletextpad=0.22,
            borderaxespad=0,
            fontsize=6.1,
        )
        fig.legend(
            handles=handles[6:],
            labels=labels[6:],
            loc="lower center",
            bbox_to_anchor=(0.5, y),
            ncol=2,
            columnspacing=0.95,
            handletextpad=0.30,
            borderaxespad=0,
            fontsize=6.2,
        )
        return
    fig.legend(
        handles=handles,
        labels=labels,
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        ncol=4 if selection else 6,
        columnspacing=0.64 if compact else 0.92,
        handletextpad=0.28 if compact else 0.38,
        borderaxespad=0,
        fontsize=6.2 if compact else mpl.rcParams["legend.fontsize"],
    )


def save_figure(fig: mpl.figure.Figure, stem: str) -> tuple[Path, Path]:
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    PDF_WORK_DIR.mkdir(parents=True, exist_ok=True)
    png = PNG_DIR / f"{stem}.png"
    pdf = PDF_WORK_DIR / f"{stem}.pdf"
    metadata = {
        "Creator": "Graph Specialisation and Metrics",
        "Title": stem,
        "Subject": "Synthetic-data figure redesign mockup",
    }
    fig.savefig(png, dpi=300, facecolor=PAPER, bbox_inches=None)
    fig.savefig(pdf, dpi=300, facecolor=PAPER, bbox_inches=None, metadata=metadata)
    plt.close(fig)
    return png, pdf


def generate_head_points(seed: int = 3407) -> list[HeadPoint]:
    """Create plausible, correlated head-level data for mock rendering."""

    rng = np.random.default_rng(seed)
    role_templates = np.array([0.82, 0.63, -0.78, -0.58, 0.05, -0.03, 0.31, -0.22])
    activity_templates = np.array([0.75, 0.30, 0.66, 0.22, 1.05, -0.28, 0.44, -0.60])
    points: list[HeadPoint] = []
    for layer in range(3):
        for run_seed in range(3):
            for head in range(8):
                role = np.clip(
                    role_templates[head]
                    + 0.08 * (layer - 1)
                    + rng.normal(0.0, 0.075),
                    -0.96,
                    0.96,
                )
                log_activity = (
                    -1.12
                    + 0.82 * layer
                    + activity_templates[head]
                    + rng.normal(0.0, 0.11)
                )
                joint = 10.0 ** log_activity
                semantic_norm = joint * (1.0 + role)
                structural_norm = joint * (1.0 - role)
                scale = 0.73 + 0.10 * run_seed + rng.normal(0.0, 0.025)
                semantic_score = max(semantic_norm * scale, 2.0e-4)
                structural_score = max(structural_norm / scale, 2.0e-4)
                if joint < 0.5:
                    family = "inert"
                elif role >= 0.20:
                    family = "semantic"
                elif role <= -0.20:
                    family = "structural"
                else:
                    family = "generalist"
                points.append(
                    HeadPoint(
                        layer=layer,
                        seed=run_seed,
                        head=head,
                        structural_score=float(structural_score),
                        semantic_score=float(semantic_score),
                        joint_sensitivity=float(joint),
                        selectivity=float(role),
                        family=family,
                    )
                )
    return points


def _scatter_heads(
    ax: mpl.axes.Axes,
    points: Sequence[HeadPoint],
    x_values: Sequence[float],
    y_values: Sequence[float],
    *,
    dim_unreliable: bool = False,
    selected_rings: bool = False,
) -> None:
    for point, x, y in zip(points, x_values, y_values):
        reliable = point.joint_sensitivity >= 0.5
        if selected_rings and point.family in {"semantic", "structural"} and point.head in {0, 2}:
            edge = SEMANTIC if point.family == "semantic" else STRUCTURAL
            edge_width = 1.45
        else:
            edge = AXIS if (reliable or not dim_unreliable) else "none"
            edge_width = 0.32
        alpha = 0.88 if (reliable or not dim_unreliable) else 0.16
        ax.scatter(
            x,
            y,
            s=31,
            marker=SEED_MARKERS[point.seed],
            facecolor=LAYER_COLOURS[point.layer],
            edgecolor=edge,
            linewidth=edge_width,
            alpha=alpha,
            zorder=3,
        )


def figure_1(points: Sequence[HeadPoint]) -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=SINGLE_SIZE)
    x = np.array([point.structural_score for point in points])
    y = np.array([point.semantic_score for point in points])
    _scatter_heads(ax, points, x, y, selected_rings=True)
    lo = 10.0 ** math.floor(math.log10(min(x.min(), y.min())))
    hi = 10.0 ** math.ceil(math.log10(max(x.max(), y.max())))
    ax.plot(
        [lo, hi],
        [lo, hi],
        color=REFERENCE,
        linewidth=0.9,
        linestyle=(0, (3.0, 2.2)),
        zorder=1,
    )
    ax.text(
        0.965,
        0.95,
        "equal task score",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=MUTED,
        fontsize=7.2,
        rotation=34,
    )
    ax.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi))
    ax.set_xlabel(r"Raw structural score $S_{\mathrm{str}}$")
    ax.set_ylabel(r"Raw semantic score $S_{\mathrm{sem}}$")
    ax.set_box_aspect(1.0)
    style_axes(ax, xgrid=True, ygrid=True, log_minor_grid=False)
    add_head_key(fig, selection=True, compact=True, y=0.012)
    fig.subplots_adjust(left=0.19, right=0.975, bottom=0.285, top=0.97)
    return save_figure(fig, "fig1_specialisation_plane_redesign")


def figure_2(points: Sequence[HeadPoint]) -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=SINGLE_SIZE)
    x = np.array([point.selectivity for point in points])
    y = np.array([point.joint_sensitivity for point in points])

    # Three faint regions make the classification rule visible without turning
    # the background into a decorative colour field.
    ax.axvspan(-1.0, -0.20, color=STRUCTURAL, alpha=0.055, linewidth=0)
    ax.axvspan(-0.20, 0.20, color=INERT, alpha=0.070, linewidth=0)
    ax.axvspan(0.20, 1.0, color=SEMANTIC, alpha=0.050, linewidth=0)
    _scatter_heads(ax, points, x, y)
    ax.axvline(0.0, color=REFERENCE, linewidth=0.72, zorder=1)
    ax.axhline(
        0.5,
        color=REFERENCE,
        linewidth=0.82,
        linestyle=(0, (3.0, 2.2)),
        zorder=1,
    )
    zone_box = {"facecolor": PAPER, "edgecolor": "none", "alpha": 0.74, "pad": 0.5}
    ax.text(0.18, 0.965, "structural", transform=ax.transAxes, ha="center", va="top", fontsize=6.8, color=STRUCTURAL, bbox=zone_box, zorder=6)
    ax.text(0.50, 0.965, "generalist", transform=ax.transAxes, ha="center", va="top", fontsize=7.0, color=MUTED, bbox=zone_box, zorder=6)
    ax.text(0.82, 0.965, "semantic", transform=ax.transAxes, ha="center", va="top", fontsize=6.8, color=SEMANTIC, bbox=zone_box, zorder=6)
    ax.annotate(
        r"reliability floor $J=0.5$",
        xy=(-0.98, 0.5),
        xytext=(4, 4),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=7.1,
        color=MUTED,
        bbox={"facecolor": PAPER, "edgecolor": "none", "alpha": 0.74, "pad": 0.5},
    )
    ax.set_xlim(-1.02, 1.02)
    ax.set_yscale("log")
    ax.set_xlabel(r"Head selectivity $D_{\mathrm{rel}}$")
    ax.set_ylabel(r"Joint sensitivity $J$")
    ax.set_box_aspect(1.0)
    style_axes(ax, xgrid=False, ygrid=True)
    add_head_key(fig, compact=True, y=0.015)
    fig.subplots_adjust(left=0.19, right=0.975, bottom=0.25, top=0.97)
    return save_figure(fig, "fig2_joint_sensitivity_selectivity_redesign")


def _rankdata(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    ranks[order] = np.arange(values.size, dtype=float)
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        for group in range(unique.size):
            mask = inverse == group
            ranks[mask] = ranks[mask].mean()
    return ranks


def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    rx = _rankdata(x)
    ry = _rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def _rho_note(ax: mpl.axes.Axes, lines: Sequence[str]) -> None:
    ax.text(
        0.04,
        0.94,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.8,
        color=MUTED,
        linespacing=1.25,
    )


def figure_3(points: Sequence[HeadPoint], seed: int = 771) -> tuple[Path, Path]:
    rng = np.random.default_rng(seed)
    j = np.array([point.joint_sensitivity for point in points])
    d = np.array([point.selectivity for point in points])
    reliable = j >= 0.5
    impact = np.maximum(0.012, 0.20 * j ** 0.86 * np.exp(rng.normal(0.0, 0.40, j.size)))
    ablation_role = np.clip(0.84 * d + rng.normal(0.0, 0.18, d.size), -1.0, 1.0)
    rescue_role = 0.15 * d + rng.normal(0.0, 0.052, d.size)

    fig, axes = plt.subplots(1, 3, figsize=TRIPTYCH_SIZE)
    specs = (
        (
            j,
            impact,
            r"Joint sensitivity $J$",
            "Cross-task ablation impact",
            "Sensitivity vs impact",
            False,
        ),
        (
            d,
            ablation_role,
            r"Score selectivity $D_{\mathrm{rel}}$",
            "Ablation role contrast",
            "Selectivity vs ablation",
            True,
        ),
        (
            d,
            rescue_role,
            r"Score selectivity $D_{\mathrm{rel}}$",
            "Rescue role contrast",
            "Selectivity vs rescue",
            True,
        ),
    )

    for index, (ax, spec) in enumerate(zip(axes, specs)):
        x, y, xlabel, ylabel, title, dim = spec
        _scatter_heads(ax, points, x, y, dim_unreliable=dim)
        panel_label(ax, f"({chr(97 + index)})", title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if index == 0:
            ax.set_xscale("log")
            ax.set_yscale("log")
            rho = _spearman(x, y)
            _rho_note(ax, [rf"all: $\rho={rho:.2f}$, $n={x.size}$"])
            style_axes(ax, xgrid=False, ygrid=True)
        else:
            ax.axhline(0.0, color=REFERENCE, linewidth=0.68, zorder=1)
            ax.axvline(0.0, color=REFERENCE, linewidth=0.68, zorder=1)
            ax.set_xlim(-1.03, 1.03)
            if index == 1:
                ax.set_ylim(-1.03, 1.03)
            rho_reliable = _spearman(x[reliable], y[reliable])
            rho_all = _spearman(x, y)
            _rho_note(
                ax,
                [
                    rf"reliable: $\rho={rho_reliable:.2f}$, $n={reliable.sum()}$",
                    rf"all: $\rho={rho_all:.2f}$, $n={x.size}$",
                ],
            )
            style_axes(ax, xgrid=False, ygrid=True)

    add_head_key(fig, y=0.014)
    fig.subplots_adjust(left=0.073, right=0.992, bottom=0.285, top=0.835, wspace=0.44)
    return save_figure(fig, "fig3_causal_validation_redesign")


FAMILY_STYLE = {
    "semantic": (SEMANTIC, "o", "-", "Semantic specialists"),
    "structural": (STRUCTURAL, "s", (0, (4.0, 1.8)), "Structural specialists"),
    "generalist": (GENERALIST, "D", (0, (2.3, 1.4)), r"High-$J$ generalists"),
    "inert": (INERT, "^", (0, (1.0, 1.6)), r"Low-$J$ / inert"),
}


def _ablation_curves(
    family: str,
    task: str,
    metric: str,
    seed: int,
) -> np.ndarray:
    # Terminal effects chosen only to recreate the qualitative structure of the
    # real figure; all plotted values are made up.
    functional_terminal = {
        ("semantic", "semantic"): 9.8,
        ("semantic", "structural"): 6.3,
        ("structural", "semantic"): 2.5,
        ("structural", "structural"): 10.4,
        ("generalist", "semantic"): 5.6,
        ("generalist", "structural"): 10.8,
        ("inert", "semantic"): 2.0,
        ("inert", "structural"): 1.2,
    }
    accuracy_terminal = {
        ("semantic", "semantic"): 3.8,
        ("semantic", "structural"): 3.4,
        ("structural", "semantic"): 0.1,
        ("structural", "structural"): 4.4,
        ("generalist", "semantic"): 0.2,
        ("generalist", "structural"): 7.8,
        ("inert", "semantic"): 0.0,
        ("inert", "structural"): 0.3,
    }
    terminal = (
        functional_terminal[(family, task)]
        if metric == "functional"
        else accuracy_terminal[(family, task)]
    )
    family_index = ("semantic", "structural", "generalist", "inert").index(family)
    task_index = ("semantic", "structural").index(task)
    metric_index = ("functional", "accuracy").index(metric)
    rng = np.random.default_rng(seed + 1000 * family_index + 100 * task_index + 10 * metric_index)
    x = np.arange(4, dtype=float)
    mean_shape = (x / 3.0) ** (1.08 + rng.normal(0.0, 0.05))
    curves = []
    for run_seed in range(3):
        scale = 1.0 + rng.normal(0.0, 0.10 + 0.03 * run_seed)
        noise = rng.normal(0.0, 0.035 * max(terminal, 0.6), x.size)
        noise[0] = 0.0
        curve = np.maximum(0.0, terminal * scale * mean_shape + noise)
        curve[0] = 0.0
        curves.append(curve)
    return np.asarray(curves)


def figure_4() -> tuple[Path, Path]:
    fig, axes = plt.subplots(2, 2, figsize=QUAD_SIZE, sharex=True, sharey="row")
    tasks = ("semantic", "structural")
    metrics = (("functional", "Functional logit impact"), ("accuracy", "Accuracy drop (pp)"))
    for col, task in enumerate(tasks):
        for row, (metric, ylabel) in enumerate(metrics):
            ax = axes[row, col]
            for family, (colour, marker, linestyle, label) in FAMILY_STYLE.items():
                curves = _ablation_curves(family, task, metric, seed=711 + row * 31 + col * 17)
                x = np.arange(curves.shape[1])
                for curve in curves:
                    ax.plot(x, curve, color=colour, alpha=0.18, linewidth=0.72, linestyle=linestyle)
                mean = curves.mean(axis=0)
                lower = curves.min(axis=0)
                upper = curves.max(axis=0)
                ax.fill_between(x, lower, upper, color=colour, alpha=0.105, linewidth=0)
                ax.plot(
                    x,
                    mean,
                    color=colour,
                    marker=marker,
                    linestyle=linestyle,
                    linewidth=1.65,
                    markersize=3.4,
                    markeredgecolor=PAPER,
                    markeredgewidth=0.35,
                    label=label,
                    zorder=3,
                )
            ax.axhline(0.0, color=REFERENCE, linewidth=0.70, zorder=1)
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == 1:
                ax.set_xlabel("Jointly ablated heads")
            if row == 0:
                ax.set_title(f"{task.capitalize()} task")
            panel_index = row * 2 + col
            panel_label(ax, f"({chr(97 + panel_index)})")
            ax.set_xticks(np.arange(4))
            style_axes(ax, xgrid=False, ygrid=True)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=4,
        columnspacing=1.3,
        handlelength=2.6,
        handletextpad=0.48,
    )
    fig.subplots_adjust(left=0.105, right=0.982, bottom=0.18, top=0.91, hspace=0.24, wspace=0.23)
    return save_figure(fig, "fig4_joint_ablation_redesign")


def _heatmap(
    matrix: np.ndarray,
    standard_deviation: np.ndarray,
    *,
    stem: str,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    xlabel: str,
    ylabel: str,
    colorbar_label: str,
    ticks: Sequence[float] | None = None,
) -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=HEATMAP_SIZE)
    vmax = max(abs(float(np.nanmin(matrix))), abs(float(np.nanmax(matrix))), 1.0e-6)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = mpl.colormaps["BrBG"]
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.add_patch(
                Rectangle(
                    (col - 0.5, row - 0.5),
                    1.0,
                    1.0,
                    facecolor=cmap(norm(float(matrix[row, col]))),
                    edgecolor=PAPER,
                    linewidth=1.2,
                )
            )
    ax.set_xlim(-0.5, matrix.shape[1] - 0.5)
    ax.set_ylim(matrix.shape[0] - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(len(col_labels)), col_labels)
    ax.set_yticks(np.arange(len(row_labels)), row_labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="both", length=0, pad=5)
    ax.tick_params(axis="x", labelsize=6.9)
    ax.tick_params(axis="y", labelsize=7.1)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = float(matrix[row, col])
            red, green, blue, _ = cmap(norm(value))
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            ax.text(
                col,
                row - 0.10,
                f"{value:+.3f}",
                ha="center",
                va="center",
                fontsize=8.2,
                fontweight="semibold",
                color=INK if luminance > 0.58 else PAPER,
            )
            ax.text(
                col,
                row + 0.14,
                rf"$\pm${float(standard_deviation[row, col]):.3f}",
                ha="center",
                va="center",
                fontsize=6.4,
                color=INK if luminance > 0.58 else PAPER,
            )
    scalar = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar = fig.colorbar(
        scalar,
        ax=ax,
        orientation="horizontal",
        fraction=0.07,
        pad=0.27,
        aspect=24,
        ticks=ticks,
    )
    colorbar.set_label(colorbar_label, labelpad=2.5, fontsize=7.1)
    colorbar.outline.set_linewidth(0.55)
    colorbar.outline.set_edgecolor(AXIS)
    colorbar.ax.tick_params(labelsize=7.2, length=2.4, pad=2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(1.0, 1.035, r"mean $\pm$ SD, $n=3$", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.4, color=MUTED)
    fig.subplots_adjust(left=0.34, right=0.975, bottom=0.33, top=0.94)
    return save_figure(fig, stem)


def figure_5() -> tuple[Path, Path]:
    return _heatmap(
        np.array([[0.004, 0.052], [-0.001, 0.011]], dtype=float),
        np.array([[0.002, 0.008], [0.001, 0.004]], dtype=float),
        stem="fig5_necessity_matrix_redesign",
        row_labels=("Semantic family", "Structural family"),
        col_labels=("Semantic", "Structural"),
        xlabel="Evaluation task",
        ylabel="Ablated family",
        colorbar_label=r"Change in cross-entropy ($\Delta$CE)",
        ticks=(-0.05, 0.00, 0.05),
    )


def figure_6() -> tuple[Path, Path]:
    return _heatmap(
        np.array([[0.154, 0.066], [-0.008, 0.128]], dtype=float),
        np.array([[0.018, 0.012], [0.006, 0.015]], dtype=float),
        stem="fig6_rescue_matrix_redesign",
        row_labels=("Semantic family", "Structural family"),
        col_labels=("Semantic", "Structural"),
        xlabel="Corruption",
        ylabel="Patched family",
        colorbar_label="Mediated-effect fraction",
        ticks=(-0.15, 0.00, 0.15),
    )


def _attention_matrix(
    role: str,
    query: int,
    source: int,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    idx = np.arange(n)
    matrix = np.zeros((n, n), dtype=float)
    for destination in range(n):
        cycle_distance = np.minimum(
            (idx - destination) % n,
            (destination - idx) % n,
        )
        local = np.exp(-cycle_distance / 1.45)
        matrix[destination] = 0.06 + 0.18 * local + rng.uniform(0.0, 0.035, n)
    if role == "Semantic specialist":
        matrix[:, source] += 0.66
        matrix[query] += 0.12 * np.roll(np.eye(1, n, 0).ravel(), source)
    elif role == "Structural specialist":
        neighbours = [(source - 1) % n, source, (source + 1) % n]
        matrix[:, neighbours] += np.array([0.24, 0.54, 0.24])
    elif role == r"High-$J$ generalist":
        matrix[:, source] += 0.46
        matrix += 0.13 * np.eye(n)
    else:
        matrix += 0.22 * np.eye(n)
        matrix[:, source] += 0.16
    matrix /= matrix.sum(axis=1, keepdims=True)
    return matrix


def figure_7(seed: int = 918) -> tuple[Path, Path]:
    rng = np.random.default_rng(seed)
    n = 16
    examples = (
        ("Semantic specialist", 10, 11, 2, 1.47, +0.92),
        ("Structural specialist", 12, 2, 2, 2.12, -0.86),
        (r"High-$J$ generalist", 3, 4, 4, 2.71, +0.11),
        ("First-layer head", 1, 13, 7, 0.42, +0.71),
    )
    matrices = [
        _attention_matrix(role, query, source, n, rng)
        for role, query, source, _head, _j, _d in examples
    ]
    vmax = max(float(matrix.max()) for matrix in matrices)
    norm = mpl.colors.Normalize(vmin=0.0, vmax=vmax)
    cmap = mpl.colormaps["cividis"]

    fig = plt.figure(figsize=ATTENTION_SIZE)
    outer = fig.add_gridspec(
        2,
        2,
        left=0.04,
        right=0.988,
        bottom=0.16,
        top=0.978,
        hspace=0.13,
        wspace=0.14,
    )
    angles = np.linspace(np.pi / 2.0, np.pi / 2.0 + 2.0 * np.pi, n, endpoint=False)
    positions = np.column_stack([np.cos(angles), np.sin(angles)])
    tick_nodes = np.array([0, 7, 15])

    for index, ((role, query, source, head, j_value, d_value), matrix) in enumerate(
        zip(examples, matrices)
    ):
        tile = outer[index // 2, index % 2].subgridspec(
            2,
            2,
            height_ratios=(0.13, 1.0),
            width_ratios=(0.96, 1.08),
            hspace=0.02,
            wspace=0.12,
        )
        title_ax = fig.add_subplot(tile[0, :])
        graph_ax = fig.add_subplot(tile[1, 0])
        matrix_ax = fig.add_subplot(tile[1, 1])
        title_ax.axis("off")
        title_ax.text(
            0.0,
            0.52,
            f"({chr(97 + index)})  {role}",
            ha="left",
            va="center",
            fontsize=8.4,
            fontweight="semibold",
        )
        title_ax.text(
            1.0,
            0.52,
            f"L{2 if index < 3 else 1} H{head}",
            ha="right",
            va="center",
            fontsize=7.0,
            color=MUTED,
        )

        query_attention = matrix[query]
        for node in range(n):
            neighbour = (node + 1) % n
            graph_ax.plot(
                [positions[node, 0], positions[neighbour, 0]],
                [positions[node, 1], positions[neighbour, 1]],
                color="#B4BBC3",
                linewidth=0.72,
                zorder=1,
            )
        graph_ax.scatter(
            positions[:, 0],
            positions[:, 1],
            c=query_attention,
            cmap=cmap,
            norm=norm,
            s=57,
            edgecolors="#69727C",
            linewidths=0.45,
            zorder=3,
        )
        # Query uses a charcoal circle; source uses a grey square.  Shape and
        # line style distinguish the roles without adding another categorical hue.
        # is allowed to imply semantic/structural task identity elsewhere.
        for selected, colour, marker in ((query, QUERY, "o"), (source, SOURCE, "s")):
            graph_ax.scatter(
                [positions[selected, 0]],
                [positions[selected, 1]],
                s=105,
                marker=marker,
                facecolors="none",
                edgecolors=PAPER,
                linewidths=2.0,
                zorder=5,
            )
            graph_ax.scatter(
                [positions[selected, 0]],
                [positions[selected, 1]],
                s=105,
                marker=marker,
                facecolors="none",
                edgecolors=colour,
                linewidths=1.15,
                zorder=6,
            )
        for node, (x, y) in enumerate(positions):
            if node not in {query, source}:
                continue
            red, green, blue, _ = cmap(norm(query_attention[node]))
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            graph_ax.text(
                x,
                y,
                str(node + 1),
                ha="center",
                va="center",
                fontsize=7.0,
                fontweight="semibold",
                color=INK if luminance > 0.57 else PAPER,
                zorder=7,
            )
        graph_ax.set(xlim=(-1.20, 1.20), ylim=(-1.18, 1.18))
        graph_ax.set_aspect("equal")
        graph_ax.axis("off")
        graph_ax.set_title("")

        matrix_ax.imshow(
            matrix,
            cmap=cmap,
            norm=norm,
            interpolation="none",
            aspect="equal",
        )
        for xy, width, height, colour in (
            ((-0.5, query - 0.5), n, 1.0, QUERY),
            ((source - 0.5, -0.5), 1.0, n, SOURCE),
        ):
            matrix_ax.add_patch(Rectangle(xy, width, height, fill=False, edgecolor=PAPER, linewidth=1.35))
            linestyle = "-" if colour == QUERY else (0, (3.0, 1.8))
            matrix_ax.add_patch(Rectangle(xy, width, height, fill=False, edgecolor=colour, linewidth=0.72, linestyle=linestyle))
        matrix_ax.set_xticks(tick_nodes, [str(node + 1) for node in tick_nodes])
        matrix_ax.set_yticks(tick_nodes, [str(node + 1) for node in tick_nodes])
        matrix_ax.tick_params(axis="both", labelsize=7.0, length=1.8, pad=1.2)
        matrix_ax.set_xlabel(
            "Source node" if index >= 2 else "",
            fontsize=7.0,
            labelpad=1.5,
        )
        matrix_ax.set_ylabel(
            "Destination node" if index % 2 == 0 else "",
            fontsize=7.0,
            labelpad=1.5,
        )
        matrix_ax.set_title("")
        for spine in matrix_ax.spines.values():
            spine.set_visible(True)
            spine.set_color(AXIS)
            spine.set_linewidth(0.55)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="none",
            markeredgecolor=QUERY,
            markeredgewidth=1.6,
            markersize=5.7,
            label="Query row",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="none",
            markerfacecolor="none",
            markeredgecolor=SOURCE,
            markeredgewidth=1.6,
            markersize=5.7,
            label="Source column",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower left",
        bbox_to_anchor=(0.04, 0.025),
        ncol=2,
        columnspacing=1.0,
        handletextpad=0.4,
    )
    colorbar_ax = fig.add_axes([0.70, 0.058, 0.275, 0.015])
    colorbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=colorbar_ax,
        orientation="horizontal",
    )
    colorbar.ax.tick_params(labelsize=5.9, length=1.8, pad=1.1)
    colorbar.ax.set_title("Attention weight", fontsize=6.7, color=MUTED, pad=2.5)
    colorbar.outline.set_linewidth(0.5)
    return save_figure(fig, "fig7_attention_visualisations_redesign")


def combine_pdfs(pdfs: Iterable[Path], output: Path) -> None:
    from pypdf import PdfWriter

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = PdfWriter()
    for pdf in pdfs:
        writer.append(str(pdf))
    writer.add_metadata(
        {
            "/Title": "Synthetic experiment figure redesign suite",
            "/Author": "Graph Specialisation and Metrics",
            "/Subject": "Deterministic synthetic-data visual mockups",
        }
    )
    with output.open("wb") as stream:
        writer.write(stream)


def make_contact_sheet(pngs: Sequence[Path], output: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    labels = (
        "1  Raw specialisation plane",
        "2  Joint sensitivity and selectivity",
        "3  Causal validation",
        "4  Joint ablation tests",
        "5  Necessity matrix",
        "6  Rescue matrix",
        "7  Attention visualisations",
    )
    thumb_width = 980
    label_height = 48
    gutter = 34
    tiles: list[Image.Image] = []
    for path, label in zip(pngs, labels):
        source = Image.open(path).convert("RGB")
        scale = thumb_width / source.width
        thumb = source.resize(
            (thumb_width, round(source.height * scale)),
            Image.Resampling.LANCZOS,
        )
        tile = Image.new("RGB", (thumb_width, thumb.height + label_height), "white")
        tile.paste(thumb, (0, label_height))
        draw = ImageDraw.Draw(tile)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 22)
        except OSError:
            font = ImageFont.load_default()
        draw.text((10, 10), label, fill=INK, font=font)
        tiles.append(tile)

    rows: list[Image.Image] = []
    for start in range(0, len(tiles), 2):
        pair = tiles[start : start + 2]
        row_height = max(tile.height for tile in pair)
        row = Image.new("RGB", (2 * thumb_width + gutter, row_height), "#EEF0F2")
        row.paste(pair[0], (0, 0))
        if len(pair) == 2:
            row.paste(pair[1], (thumb_width + gutter, 0))
        rows.append(row)
    sheet = Image.new(
        "RGB",
        (2 * thumb_width + gutter, sum(row.height for row in rows) + gutter * (len(rows) - 1)),
        "#EEF0F2",
    )
    top = 0
    for row in rows:
        sheet.paste(row, (0, top))
        top += row.height + gutter
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, dpi=(180, 180), quality=95)


def make_layout_preview(pngs: Sequence[Path], output: Path) -> None:
    """Compose the figures at their intended half/full-width relationships."""

    from PIL import Image

    images = [Image.open(path).convert("RGB") for path in pngs]
    full_width = 1600
    gutter = 48
    section_gap = 36
    half_width = (full_width - gutter) // 2

    def fit_width(image: Image.Image, width: int) -> Image.Image:
        return image.resize(
            (width, round(image.height * width / image.width)),
            Image.Resampling.LANCZOS,
        )

    rows: list[Image.Image] = []
    for indices in ((0, 1), (2,), (3,), (4, 5), (6,)):
        if len(indices) == 2:
            pair = [fit_width(images[index], half_width) for index in indices]
            row_height = max(image.height for image in pair)
            row = Image.new("RGB", (full_width, row_height), PAPER)
            row.paste(pair[0], (0, 0))
            row.paste(pair[1], (half_width + gutter, 0))
        else:
            row = fit_width(images[indices[0]], full_width)
        rows.append(row)

    canvas = Image.new(
        "RGB",
        (
            full_width,
            sum(row.height for row in rows) + section_gap * (len(rows) - 1),
        ),
        "#E8EBEE",
    )
    top = 0
    for row in rows:
        canvas.paste(row, (0, top))
        top += row.height + section_gap
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, dpi=(180, 180), quality=95)


def main() -> None:
    configure_matplotlib()
    points = generate_head_points()
    outputs = [
        figure_1(points),
        figure_2(points),
        figure_3(points),
        figure_4(),
        figure_5(),
        figure_6(),
        figure_7(),
    ]
    pngs = [png for png, _pdf in outputs]
    pdfs = [pdf for _png, pdf in outputs]
    combine_pdfs(pdfs, FINAL_PDF)
    make_contact_sheet(pngs, CONTACT_SHEET)
    make_layout_preview(pngs, LAYOUT_PREVIEW)
    print(f"Wrote {len(pngs)} PNG previews to {PNG_DIR}")
    print(f"Wrote review PDF to {FINAL_PDF}")
    print(f"Wrote contact sheet to {CONTACT_SHEET}")
    print(f"Wrote intended-layout preview to {LAYOUT_PREVIEW}")


if __name__ == "__main__":
    main()
