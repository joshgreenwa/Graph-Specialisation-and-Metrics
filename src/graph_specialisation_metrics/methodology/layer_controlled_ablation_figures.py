"""Dissertation-matched figures for the layer-controlled ablation correction.

The raw-coordinate companion figures intentionally preserve the layer colours,
marker identities, axes, and export canvases from Figures 4.5(a) and 5.6.  The
primary corrected figures use within-(trained-seed, layer) percentile ranks on
both axes.  This makes the plotted coordinates, as well as the inset statistic,
insensitive to between-layer location shifts.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .cache import atomic_json
from .figures import FigureTheme, publication_style

SYNTHETIC_FIGSIZE = (2.1766666666666667, 2.36)
MOLECULAR_FIGSIZE = (4.2, 3.55)
PNG_DPI = 600
PDF_RASTER_DPI = 1200
SYNTHETIC_PDF_RASTER_DPI = 600

CoordinateView = Literal["raw", "within_layer_percentile_ranks"]
RAW_COORDINATES: CoordinateView = "raw"
WITHIN_LAYER_PERCENTILE_RANKS: CoordinateView = "within_layer_percentile_ranks"
PERCENTILE_RANK_DEFINITION = "(average_rank - 0.5) / stratum_size"

SYNTHETIC_LAYER_COLOURS = ("#6B5B95", "#2A8C82", "#B88720")
SEED_MARKERS = ("o", "s", "^", "D", "P", "X", "v", "<", ">")

INK_COLOUR = "#252A31"
AXIS_COLOUR = "#3D434B"
GRID_COLOUR = "#E7EAEE"
PAPER_COLOUR = "#FFFFFF"


def molecular_figure_theme() -> FigureTheme:
    """Return the exact Figure 5.6 theme used by the dissertation renderer."""

    return FigureTheme(
        width=MOLECULAR_FIGSIZE[0],
        height=MOLECULAR_FIGSIZE[1],
        dpi=PNG_DPI,
        font_size=9.5,
        label_size=10.0,
        title_size=10.5,
        tick_size=8.5,
        marker_size=32.0,
        line_width=1.2,
        grid_alpha=0.14,
    )


def _save_exact_canvas(
    figure: Any,
    output_base: Path,
    *,
    metadata: Mapping[str, Any],
    theme: FigureTheme,
    subject: str,
    pdf_raster_dpi: int = PDF_RASTER_DPI,
) -> tuple[Path, Path, Path]:
    """Save a vector-first PDF, 600-DPI PNG, and provenance sidecar."""

    import matplotlib as mpl
    import matplotlib.pyplot as plt

    output_base.parent.mkdir(parents=True, exist_ok=True)
    png = output_base.with_suffix(".png")
    pdf = output_base.with_suffix(".pdf")
    sidecar = output_base.with_suffix(".metadata.json")
    # Interactive backends may quantise an explicit figsize through an integer-pixel
    # window manager.  Reset it without forwarding immediately before export so the
    # dissertation's fractional-inch PDF MediaBox is backend-independent.
    figure.set_size_inches(theme.width, theme.height, forward=False)
    settings = {
        "font.family": theme.font_family,
        "pdf.fonttype": 42,
        "pdf.use14corefonts": False,
        "pdf.compression": 9,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "path.simplify": False,
        "agg.path.chunksize": 0,
        "image.composite_image": False,
        "image.interpolation": "none",
        "savefig.transparent": False,
    }
    with mpl.rc_context(settings):
        figure.savefig(
            png,
            dpi=PNG_DPI,
            facecolor=PAPER_COLOUR,
            bbox_inches=None,
            transparent=False,
        )
        figure.savefig(
            pdf,
            dpi=pdf_raster_dpi,
            facecolor=PAPER_COLOUR,
            bbox_inches=None,
            transparent=False,
            metadata={
                "Creator": "Graph Specialisation and Metrics",
                "Title": output_base.name,
                "Subject": subject,
            },
        )
    atomic_json(
        sidecar,
        {
            "figure": output_base.name,
            "theme": dataclasses.asdict(theme),
            "canvas_inches": [float(figure.get_figwidth()), float(figure.get_figheight())],
            "png_dpi": PNG_DPI,
            "pdf_raster_fallback_dpi": int(pdf_raster_dpi),
            "vector_first": True,
            **dict(metadata),
        },
    )
    plt.close(figure)
    return png, pdf, sidecar


def _synthetic_style() -> None:
    import matplotlib as mpl

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
            "axes.labelcolor": INK_COLOUR,
            "axes.edgecolor": AXIS_COLOUR,
            "axes.linewidth": 0.72,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 7.7,
            "ytick.labelsize": 7.7,
            "xtick.color": AXIS_COLOUR,
            "ytick.color": AXIS_COLOUR,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.minor.size": 1.8,
            "ytick.minor.size": 1.8,
            "text.color": INK_COLOUR,
            "figure.facecolor": PAPER_COLOUR,
            "axes.facecolor": PAPER_COLOUR,
            "savefig.facecolor": PAPER_COLOUR,
            "pdf.fonttype": 42,
            "pdf.use14corefonts": False,
            "pdf.compression": 9,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "path.simplify": False,
            "agg.path.chunksize": 0,
            "image.interpolation": "none",
        }
    )


def _statistic_label(estimate: float, low: float | None, high: float | None) -> str:
    label = rf"Within-layer $\bar{{\rho}}$ = {float(estimate):.2f}"
    if low is not None and high is not None and np.isfinite((low, high)).all():
        label += rf"  [{float(low):.2f}, {float(high):.2f}]"
    return label


def within_stratum_percentile_ranks(
    values: Sequence[float] | np.ndarray,
    strata: Sequence[Any],
) -> np.ndarray:
    """Return tie-aware percentile ranks calculated separately in each stratum.

    Average ranks are converted with the plotting-position convention
    ``(rank - 0.5) / n``.  Values therefore lie strictly inside ``(0, 1)`` and
    remain visible when the corrected panels use fixed zero-to-one axes.
    """

    from scipy.stats import rankdata

    array = np.asarray(values, dtype=np.float64).reshape(-1)
    stratum_values = list(strata)
    if array.size == 0:
        raise ValueError("percentile ranks need at least one value")
    if len(stratum_values) != array.size:
        raise ValueError("values and strata must have identical lengths")
    if not np.isfinite(array).all():
        raise ValueError("percentile ranks require finite values")

    positions: dict[Any, list[int]] = {}
    for position, stratum in enumerate(stratum_values):
        try:
            positions.setdefault(stratum, []).append(position)
        except TypeError as error:
            raise ValueError("percentile-rank strata must be hashable") from error

    transformed = np.empty(array.size, dtype=np.float64)
    for indices in positions.values():
        index = np.asarray(indices, dtype=np.int64)
        ranks = np.asarray(rankdata(array[index], method="average"), dtype=np.float64)
        transformed[index] = (ranks - 0.5) / float(index.size)
    return transformed


def _validate_coordinate_view(coordinate_view: str) -> CoordinateView:
    if coordinate_view not in {RAW_COORDINATES, WITHIN_LAYER_PERCENTILE_RANKS}:
        raise ValueError(
            "coordinate_view must be 'raw' or 'within_layer_percentile_ranks', "
            f"got {coordinate_view!r}"
        )
    return coordinate_view  # type: ignore[return-value]


def _view_metadata(coordinate_view: CoordinateView) -> dict[str, Any]:
    if coordinate_view == RAW_COORDINATES:
        return {
            "coordinate_view": RAW_COORDINATES,
            "presentation_change": "statistic annotation only",
            "role": "raw-coordinate dissertation companion",
        }
    return {
        "coordinate_view": WITHIN_LAYER_PERCENTILE_RANKS,
        "rank_strata": ["trained_seed", "layer"],
        "percentile_rank_definition": PERCENTILE_RANK_DEFINITION,
        "presentation_change": "both axes transformed within trained seed and layer",
        "role": "primary layer-confound-free diagnostic",
    }


def build_synthetic_dissertation_panel(
    records: Sequence[Mapping[str, Any]],
    *,
    estimate: float,
    coordinate_view: CoordinateView = RAW_COORDINATES,
) -> Any:
    """Build Figure 4.5(a) in raw or confound-free rank coordinates."""

    import matplotlib as mpl
    import matplotlib.pyplot as plt

    view = _validate_coordinate_view(coordinate_view)
    if not records:
        raise ValueError("synthetic panel needs at least one head record")
    layers = sorted({int(row["layer"]) for row in records})
    seeds = sorted({int(row["seed"]) for row in records})
    if layers != list(range(len(layers))):
        raise ValueError(f"synthetic layers must be consecutive from zero, got {layers}")
    if len(layers) > len(SYNTHETIC_LAYER_COLOURS):
        raise ValueError("the dissertation synthetic palette supports exactly three layers")
    if len(seeds) > len(SEED_MARKERS):
        raise ValueError("too many synthetic seeds for the dissertation marker palette")

    joint = np.asarray([row["joint_sensitivity"] for row in records], dtype=np.float64)
    impact = np.asarray([row["ablation_impact"] for row in records], dtype=np.float64)
    if not (np.isfinite(joint).all() and np.isfinite(impact).all()):
        raise ValueError("synthetic panels require complete finite head rows")
    strata = [(int(row["seed"]), int(row["layer"])) for row in records]
    if view == WITHIN_LAYER_PERCENTILE_RANKS:
        x_values = within_stratum_percentile_ranks(joint, strata)
        y_values = within_stratum_percentile_ranks(impact, strata)
    else:
        if not ((joint > 0.0).all() and (impact > 0.0).all()):
            raise ValueError("the dissertation synthetic log-log panel needs positive values")
        x_values = joint
        y_values = impact

    with mpl.rc_context():
        _synthetic_style()
        figure, axis = plt.subplots(figsize=SYNTHETIC_FIGSIZE)
        seed_position = {seed: position for position, seed in enumerate(seeds)}
        for position, row in enumerate(records):
            axis.scatter(
                float(x_values[position]),
                float(y_values[position]),
                s=23.5,
                marker=SEED_MARKERS[seed_position[int(row["seed"])]],
                facecolor=SYNTHETIC_LAYER_COLOURS[int(row["layer"])],
                edgecolor=PAPER_COLOUR,
                linewidth=0.45,
                alpha=0.90,
                zorder=3,
            )
        if view == RAW_COORDINATES:
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.set_xlabel(r"Joint sensitivity $J$", fontsize=8.0)
            axis.set_ylabel("Ablation impact", fontsize=8.0)
        else:
            axis.set_xscale("linear")
            axis.set_yscale("linear")
            axis.set_xlim(0.0, 1.0)
            axis.set_ylim(0.0, 1.0)
            axis.set_xlabel(r"Within-layer percentile of $J$", fontsize=8.0)
            axis.set_ylabel("Within-layer impact percentile", fontsize=8.0)
        axis.set_title(
            "(a) Sensitivity vs impact",
            loc="left",
            fontsize=7.7,
            fontweight="semibold",
            pad=4.0,
        )
        axis.text(
            0.045,
            0.955,
            _statistic_label(estimate, None, None),
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=7.5,
            color=INK_COLOUR,
            linespacing=1.22,
            bbox={
                "boxstyle": "round,pad=0.24",
                "facecolor": PAPER_COLOUR,
                "edgecolor": "#D7DCE1",
                "linewidth": 0.55,
                "alpha": 0.96,
            },
            zorder=10,
        )
        axis.grid(axis="y", which="major", color=GRID_COLOUR, linewidth=0.42, alpha=0.74)
        axis.set_axisbelow(True)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(AXIS_COLOUR)
            axis.spines[side].set_linewidth(0.72)
        figure.subplots_adjust(left=0.275, right=0.925, bottom=0.22, top=0.84)
    return figure


def _synthetic_figure_theme() -> FigureTheme:
    return FigureTheme(
        width=SYNTHETIC_FIGSIZE[0],
        height=SYNTHETIC_FIGSIZE[1],
        dpi=PNG_DPI,
        font_size=8.5,
        label_size=8.8,
        title_size=9.0,
        tick_size=7.7,
        marker_size=23.5,
    )


def render_synthetic_dissertation_panel(
    records: Sequence[Mapping[str, Any]],
    *,
    estimate: float,
    output_dir: str | Path,
    metadata: Mapping[str, Any],
    coordinate_view: CoordinateView = RAW_COORDINATES,
    output_stem: str | None = None,
) -> tuple[Path, Path, Path]:
    """Render Figure 4.5(a), retaining raw coordinates by default."""

    view = _validate_coordinate_view(coordinate_view)
    figure = build_synthetic_dissertation_panel(
        records,
        estimate=estimate,
        coordinate_view=view,
    )
    stem = output_stem or (
        "fig2a_sensitivity_impact"
        if view == RAW_COORDINATES
        else "fig2a_sensitivity_impact_within_layer_ranks"
    )
    return _save_exact_canvas(
        figure,
        Path(output_dir) / stem,
        metadata={
            "dissertation_figure": "4.5(a)",
            **dict(metadata),
            **_view_metadata(view),
        },
        theme=_synthetic_figure_theme(),
        subject="Synthetic experiment layer-controlled correction",
        pdf_raster_dpi=SYNTHETIC_PDF_RASTER_DPI,
    )


def render_synthetic_within_layer_rank_panel(
    records: Sequence[Mapping[str, Any]],
    *,
    estimate: float,
    output_dir: str | Path,
    metadata: Mapping[str, Any],
    output_stem: str = "fig2a_sensitivity_impact_within_layer_ranks",
) -> tuple[Path, Path, Path]:
    """Render the primary synthetic plot in confound-free rank coordinates."""

    return render_synthetic_dissertation_panel(
        records,
        estimate=estimate,
        output_dir=output_dir,
        metadata=metadata,
        coordinate_view=WITHIN_LAYER_PERCENTILE_RANKS,
        output_stem=output_stem,
    )


def _style_molecular_axis(axis: Any) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", alpha=0.14, linewidth=0.55)
    axis.set_axisbelow(True)


def build_molecular_dissertation_panel(
    record: Mapping[str, Any],
    *,
    estimate: float,
    low: float | None,
    high: float | None,
    coordinate_view: CoordinateView = RAW_COORDINATES,
) -> Any:
    """Build one Figure 5.6 component in raw or confound-free coordinates."""

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    view = _validate_coordinate_view(coordinate_view)
    joint = np.asarray(record["joint_sensitivity"], dtype=np.float64).reshape(-1)
    impact = np.asarray(record["ablation_impact"], dtype=np.float64).reshape(-1)
    layers = np.asarray(record["layer"], dtype=np.int64).reshape(-1)
    if not (joint.shape == impact.shape == layers.shape):
        raise ValueError("molecular scatter arrays must have identical shapes")
    finite = np.isfinite(joint) & np.isfinite(impact) & np.isfinite(layers)
    if not finite.all():
        raise ValueError("molecular dissertation panels require complete finite head rows")
    if joint.size == 0:
        raise ValueError("molecular panel needs at least one head record")
    if np.min(layers) < 0:
        raise ValueError("molecular layer indices must be non-negative")
    maximum_layer = int(np.max(layers))
    layer_count = maximum_layer + 1
    theme = molecular_figure_theme()
    seed = int(record["seed"])
    if view == WITHIN_LAYER_PERCENTILE_RANKS:
        strata = [(seed, int(layer)) for layer in layers]
        x_values = within_stratum_percentile_ranks(joint, strata)
        y_values = within_stratum_percentile_ranks(impact, strata)
    else:
        x_values = joint
        y_values = impact

    with publication_style(theme):
        figure = plt.figure(figsize=MOLECULAR_FIGSIZE)
        axis = figure.add_axes((0.17, 0.25, 0.56, 0.66))
        colorbar_axis = figure.add_axes((0.80, 0.25, 0.035, 0.66))
        cmap = plt.get_cmap("viridis", layer_count)
        norm = mpl.colors.BoundaryNorm(
            np.arange(-0.5, layer_count + 0.5, 1.0),
            cmap.N,
        )
        axis.scatter(
            x_values,
            y_values,
            c=layers,
            cmap=cmap,
            norm=norm,
            marker=SEED_MARKERS[0],
            s=theme.marker_size * 0.54,
            alpha=0.86,
            linewidths=0.22,
            edgecolors="white",
            zorder=2,
        )
        if view == RAW_COORDINATES:
            axis.set_xlabel(r"Joint sensitivity, $J$")
            axis.set_ylabel("Head-ablation impact")
            axis.set_xlim(left=0.0)
            axis.set_ylim(bottom=0.0)
        else:
            axis.set_xlabel(r"Within-layer percentile of $J$")
            axis.set_ylabel("Within-layer impact percentile")
            axis.set_xlim(0.0, 1.0)
            axis.set_ylim(0.0, 1.0)
        axis.set_title("Joint sensitivity and head-ablation impact")
        _style_molecular_axis(axis)
        layer_ticks = np.unique(
            np.rint(np.linspace(0, maximum_layer, min(layer_count, 6))).astype(int)
        )
        colorbar = figure.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=colorbar_axis,
            ticks=layer_ticks,
        )
        colorbar.set_label("Layer")
        colorbar.outline.set_linewidth(0.55)
        colorbar.ax.tick_params(length=2.5, width=0.55)
        axis.text(
            0.025,
            0.975,
            _statistic_label(estimate, low, high),
            transform=axis.transAxes,
            ha="left",
            va="top",
            color="black",
            fontsize=theme.tick_size,
            bbox={
                "boxstyle": "square,pad=0.24",
                "facecolor": "white",
                "edgecolor": "#C8C8C8",
                "linewidth": 0.5,
                "alpha": 0.96,
            },
            zorder=4,
        )
        handle = Line2D(
            [0],
            [0],
            color=theme.central_color,
            marker=SEED_MARKERS[0],
            linestyle="none",
            markerfacecolor="none",
            markeredgewidth=0.9,
            label=f"Seed {seed}",
        )
        figure.legend(
            handles=[handle],
            frameon=False,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=4,
            columnspacing=1.15,
            handletextpad=0.45,
            borderaxespad=0,
        )

    return figure


def render_molecular_dissertation_panel(
    record: Mapping[str, Any],
    *,
    estimate: float,
    low: float | None,
    high: float | None,
    output_dir: str | Path,
    metadata: Mapping[str, Any],
    coordinate_view: CoordinateView = RAW_COORDINATES,
    output_stem: str | None = None,
) -> tuple[Path, Path, Path]:
    """Render one Figure 5.6 panel, retaining raw coordinates by default."""

    view = _validate_coordinate_view(coordinate_view)
    figure = build_molecular_dissertation_panel(
        record,
        estimate=estimate,
        low=low,
        high=high,
        coordinate_view=view,
    )
    stem = output_stem or (
        "01_joint_sensitivity_head_ablation"
        if view == RAW_COORDINATES
        else "01_joint_sensitivity_head_ablation_within_layer_ranks"
    )
    return _save_exact_canvas(
        figure,
        Path(output_dir) / stem,
        metadata={
            "dissertation_figure": "5.6",
            **dict(metadata),
            **_view_metadata(view),
        },
        theme=molecular_figure_theme(),
        subject="Molecular head-ablation layer-controlled correction",
    )


def render_molecular_within_layer_rank_panel(
    record: Mapping[str, Any],
    *,
    estimate: float,
    low: float | None,
    high: float | None,
    output_dir: str | Path,
    metadata: Mapping[str, Any],
    output_stem: str = "01_joint_sensitivity_head_ablation_within_layer_ranks",
) -> tuple[Path, Path, Path]:
    """Render the primary molecular plot in confound-free rank coordinates."""

    return render_molecular_dissertation_panel(
        record,
        estimate=estimate,
        low=low,
        high=high,
        output_dir=output_dir,
        metadata=metadata,
        coordinate_view=WITHIN_LAYER_PERCENTILE_RANKS,
        output_stem=output_stem,
    )


__all__ = [
    "MOLECULAR_FIGSIZE",
    "PNG_DPI",
    "RAW_COORDINATES",
    "SYNTHETIC_FIGSIZE",
    "SYNTHETIC_PDF_RASTER_DPI",
    "WITHIN_LAYER_PERCENTILE_RANKS",
    "CoordinateView",
    "build_molecular_dissertation_panel",
    "build_synthetic_dissertation_panel",
    "molecular_figure_theme",
    "render_molecular_dissertation_panel",
    "render_molecular_within_layer_rank_panel",
    "render_synthetic_dissertation_panel",
    "render_synthetic_within_layer_rank_panel",
    "within_stratum_percentile_ranks",
]
