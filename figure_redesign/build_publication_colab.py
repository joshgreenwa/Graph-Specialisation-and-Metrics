"""Build the publication-ready, single-cell Colab notebook.

This transformer preserves the experiment and analysis implementation from the
original dissertation notebook.  It replaces only the publication renderer,
output documentation, and figure summary metadata, then removes stale outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_SOURCE = Path("/Users/joshgreen/Downloads/dissertation_final_mixed_synthetic.ipynb")
DEFAULT_OUTPUT = Path(
    "/Users/joshgreen/Documents/Graph Specialisation and Metrics/output/notebooks/"
    "dissertation_final_mixed_synthetic_publication.ipynb"
)


PUBLICATION_CONSTANTS = r'''FIGURE_VERSION = "mixed-task-publication-harmonised-v11"
ATTENTION_FIGURE_VERSION = "mixed-task-attention-publication-v8"
FAMILY_ABLATION_REVISION = "score-selected-prefix-family-ablation-v1"
DJ_FAMILY_ABLATION_REVISION = "joint-selectivity-prefix-family-ablation-v1"
OFFICIAL_GRIT_URL = "https://github.com/LiamMa/GRIT.git"
OFFICIAL_GRIT_COMMIT = "6c988ea600a606fbb49a2246c64a2d37396b3ab5"
DEFAULT_GRIT_DIR = "/content/GRIT"
DEFAULT_DRIVE_ROOT = "/content/drive/MyDrive/graph_specialisation_metrics/causal_specialisation_double_dissociation"
MODE_SEMANTIC = 0
MODE_STRUCTURAL = 1
MODE_NAMES = {MODE_SEMANTIC: "semantic", MODE_STRUCTURAL: "structural"}
EPS = 1.0e-12
DJ_RELIABILITY_FLOOR = 0.50  # combined sensitivity relative to the within-seed head mean
DJ_SELECTIVITY_THRESHOLD = 0.20  # equivalent to a 1.5:1 calibrated channel ratio

# Manuscript geometry: components in a row have matched canvases and margins.
TEXT_WIDTH = 6.85
PAIR_GAP = 0.22
TRIPTYCH_GAP = 0.16
HALF_WIDTH = (TEXT_WIDTH - PAIR_GAP) / 2.0
THIRD_WIDTH = (TEXT_WIDTH - 2.0 * TRIPTYCH_GAP) / 3.0
PAIR_FIGSIZE = (HALF_WIDTH, 3.12)
CAUSAL_FIGSIZE = (THIRD_WIDTH, 2.36)
ABLATION_FIGSIZE = (TEXT_WIDTH, 4.25)
MATRIX_FIGSIZE = (HALF_WIDTH, 2.70)
ATTENTION_FIGSIZE = (TEXT_WIDTH, 4.75)
PNG_DPI = 600
PDF_RASTER_DPI = 600
INLINE_COMPONENT_WIDTH = 850
INLINE_FULL_WIDTH = 1100
RAW_SCORE_DISPLAY_FLOOR = 1.0e-4

# One restrained visual language across the complete synthetic section.
SEMANTIC_COLOR = "#D55E00"
STRUCTURAL_COLOR = "#0072B2"
LAYER_COLORS = ("#6B5B95", "#2A8C82", "#B88720")
GENERALIST_COLOR = "#50555A"
INERT_COLOR = "#A2A7AC"
INK_COLOR = "#252A31"
MUTED_COLOR = "#68717A"
AXIS_COLOR = "#3D434B"
GRID_COLOR = "#E7EAEE"
REFERENCE_COLOR = "#8E969E"
PAPER_COLOR = "#FFFFFF"
QUERY_COLOR = "#20252B"
SOURCE_COLOR = "#949CA4"
NEUTRAL_BAND_COLOR = "#F0F1F2"
EFFECT_CMAP_COLORS = ("#3B75AF", "#F7F7F5", "#C84E3A")
ATTENTION_EXAMPLE_GRAPHS = 64
ATTENTION_ROW_KEYS = ("semantic", "structural", "generalist", "layer0")
'''


PUBLICATION_HELPERS = r'''def configure_matplotlib() -> Any:
    """Apply portable, Colab-safe publication defaults."""

    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "font.weight": "regular",
        "mathtext.fontset": "dejavusans",
        "axes.titlesize": 9.0,
        "axes.titleweight": "semibold",
        "axes.titlepad": 5.0,
        "axes.labelsize": 8.8,
        "axes.labelcolor": INK_COLOR,
        "axes.edgecolor": AXIS_COLOR,
        "axes.linewidth": 0.72,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 7.7,
        "ytick.labelsize": 7.7,
        "xtick.color": AXIS_COLOR,
        "ytick.color": AXIS_COLOR,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.size": 1.8,
        "ytick.minor.size": 1.8,
        "legend.fontsize": 7.4,
        "legend.title_fontsize": 7.4,
        "legend.frameon": False,
        "text.color": INK_COLOR,
        "figure.facecolor": PAPER_COLOR,
        "axes.facecolor": PAPER_COLOR,
        "savefig.facecolor": PAPER_COLOR,
        "figure.dpi": 200,
        "savefig.dpi": PDF_RASTER_DPI,
        "savefig.bbox": None,
        "savefig.transparent": False,
        "pdf.fonttype": 42,
        "pdf.use14corefonts": False,
        "pdf.compression": 9,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "path.simplify": False,
        "agg.path.chunksize": 0,
        "image.interpolation": "none",
        "axes.unicode_minus": True,
    })
    return plt


def save_figure(fig: Any, base: Path) -> list[str]:
    """Write a 600-ppi preview and a vector-first publication PDF."""

    import matplotlib.pyplot as plt

    base.parent.mkdir(parents=True, exist_ok=True)
    png = base.with_suffix(".png")
    pdf = base.with_suffix(".pdf")
    metadata = {
        "Creator": "Graph Specialisation and Metrics",
        "Title": base.stem,
        "Subject": "Synthetic experiment publication figure",
    }
    fig.savefig(
        png,
        dpi=PNG_DPI,
        facecolor=PAPER_COLOR,
        bbox_inches=None,
        transparent=False,
    )
    fig.savefig(
        pdf,
        dpi=PDF_RASTER_DPI,
        facecolor=PAPER_COLOR,
        bbox_inches=None,
        transparent=False,
        metadata=metadata,
    )
    print(f"[figure] {png}\n[figure] {pdf}", flush=True)
    # Show review-sized PNGs in Colab without changing manuscript PDF geometry.
    backend = str(plt.get_backend()).lower()
    if "inline" in backend:
        from IPython.display import Image, display

        preview_width = (
            INLINE_FULL_WIDTH
            if fig.get_figwidth() >= 0.9 * TEXT_WIDTH
            else INLINE_COMPONENT_WIDTH
        )
        plt.close(fig)
        display(Image(filename=str(png), width=preview_width))
    else:
        if "agg" not in backend:
            plt.show()
        plt.close(fig)
    return [str(png), str(pdf)]


def layer_palette(layers: int) -> list[Any]:
    """Return the approved layer palette, with a safe fallback for non-default runs."""

    import matplotlib.pyplot as plt

    if layers <= len(LAYER_COLORS):
        return list(LAYER_COLORS[:layers])
    fallback = plt.get_cmap("tab10")
    return list(LAYER_COLORS) + [
        fallback(index % 10)
        for index in range(layers - len(LAYER_COLORS))
    ]


SEED_MARKERS = ("o", "s", "^", "D", "P")


def head_legend_handles(
    cfg: Config,
    seeds: Sequence[int],
) -> tuple[list[Any], list[Any]]:
    from matplotlib.lines import Line2D

    colors = layer_palette(cfg.layers)
    layer_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=colors[layer],
            markeredgecolor=AXIS_COLOR,
            markeredgewidth=0.35,
            markersize=5.1,
            label=f"L{layer + 1}",
        )
        for layer in range(cfg.layers)
    ]
    seed_handles = [
        Line2D(
            [0],
            [0],
            marker=SEED_MARKERS[index % len(SEED_MARKERS)],
            linestyle="none",
            markerfacecolor=PAPER_COLOR,
            markeredgecolor=MUTED_COLOR,
            markeredgewidth=0.8,
            markersize=5.0,
            label=f"seed {seed}",
        )
        for index, seed in enumerate(seeds)
    ]
    return layer_handles, seed_handles


def add_head_legend(
    fig: Any,
    cfg: Config,
    seeds: Sequence[int],
    *,
    y: float = 0.045,
) -> None:
    layer_handles, seed_handles = head_legend_handles(cfg, seeds)
    handles = layer_handles + seed_handles
    fig.legend(
        handles,
        [handle.get_label() for handle in handles],
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        ncol=max(1, len(handles)),
        columnspacing=0.48,
        handletextpad=0.24,
        borderaxespad=0,
        fontsize=7.0,
    )


def add_inset_head_legends(
    ax: Any,
    cfg: Config,
    seeds: Sequence[int],
) -> None:
    """Legacy-compatible compact inset legends for inactive exploratory plots."""

    layer_handles, seed_handles = head_legend_handles(cfg, seeds)
    layer_legend = ax.legend(
        handles=layer_handles,
        loc="upper left",
        frameon=False,
        borderpad=0.2,
        handletextpad=0.3,
    )
    ax.add_artist(layer_legend)
    ax.legend(
        handles=seed_handles,
        loc="lower right",
        frameon=False,
        borderpad=0.2,
        handletextpad=0.3,
    )


def selection_legend_handles() -> tuple[list[Any], list[str]]:
    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=PAPER_COLOR,
            markeredgecolor=color,
            markeredgewidth=1.5,
            markersize=5.8,
        )
        for color in (SEMANTIC_COLOR, STRUCTURAL_COLOR)
    ]
    return handles, ["semantic selection", "structural selection"]


def style_scatter_axes(
    ax: Any,
    *,
    zero_lines: bool = False,
    xgrid: bool = False,
    ygrid: bool = True,
    log_minor_grid: bool = False,
) -> None:
    """Apply the shared restrained grid and spine treatment."""

    if zero_lines:
        ax.axhline(0.0, color=REFERENCE_COLOR, linewidth=0.62, zorder=1)
        ax.axvline(0.0, color=REFERENCE_COLOR, linewidth=0.62, zorder=1)
    if xgrid:
        ax.grid(axis="x", which="major", color=GRID_COLOR, linewidth=0.42, alpha=0.74)
    if ygrid:
        ax.grid(axis="y", which="major", color=GRID_COLOR, linewidth=0.42, alpha=0.74)
    if log_minor_grid:
        ax.grid(axis="both", which="minor", color=GRID_COLOR, linewidth=0.34, alpha=0.35)
    ax.set_axisbelow(True)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS_COLOR)
        ax.spines[side].set_linewidth(0.72)


def publication_log_limits(*arrays: Any) -> tuple[float, float]:
    """Return padded shared log limits containing every displayed observation."""

    values = np.concatenate([
        np.asarray(array, dtype=float).ravel()
        for array in arrays
    ])
    if not (
        values.size
        and np.all(np.isfinite(values))
        and np.all(values > 0.0)
    ):
        raise ValueError("shared log limits require positive finite values")
    logs = np.log10(values)
    span = max(float(logs.max() - logs.min()), 1.0)
    padding = max(0.08, 0.06 * span)
    return (
        10.0 ** float(logs.min() - padding),
        10.0 ** float(logs.max() + padding),
    )


def raw_score_display_mask(
    structural: np.ndarray,
    semantic: np.ndarray,
) -> np.ndarray:
    """Hide heads below the approved raw-score display floor in either channel."""

    structural = np.asarray(structural, dtype=float)
    semantic = np.asarray(semantic, dtype=float)
    if structural.shape != semantic.shape:
        raise ValueError("raw-score arrays must have matching shapes")
    if not (
        np.all(np.isfinite(structural))
        and np.all(np.isfinite(semantic))
        and np.all(structural >= 0.0)
        and np.all(semantic >= 0.0)
    ):
        raise ValueError("raw scores must be non-negative and finite")

    return (
        (structural >= RAW_SCORE_DISPLAY_FLOOR)
        & (semantic >= RAW_SCORE_DISPLAY_FLOOR)
    )


'''


RAW_SCORE_FIGURE = r'''def figure_specialisation_plane(
    results: Sequence[dict[str, Any]],
    cfg: Config,
    figures_dir: Path,
) -> list[str]:
    """Raw task scores on the matched left-hand half-width canvas."""

    plt = configure_matplotlib()
    fig, ax = plt.subplots(figsize=PAIR_FIGSIZE)
    colors = layer_palette(cfg.layers)
    structural_cube = np.stack([
        np.asarray(result["structural_score"], dtype=float)
        for result in results
    ])
    semantic_cube = np.stack([
        np.asarray(result["semantic_score"], dtype=float)
        for result in results
    ])
    keep = raw_score_display_mask(structural_cube, semantic_cube)
    omitted_indices = np.argwhere(~keep)
    if len(omitted_indices):
        print(
            f"[figure] fig1a display-only omissions below {RAW_SCORE_DISPLAY_FLOOR:.0e}: "
            f"{len(omitted_indices)} head(s)",
            flush=True,
        )
        for result_index, layer, head in omitted_indices:
            result_index, layer, head = map(int, (result_index, layer, head))
            print(
                f"  seed {int(results[result_index]['seed'])}, "
                f"L{layer + 1} H{head + 1}, "
                f"structural={structural_cube[result_index, layer, head]:.3e}, "
                f"semantic={semantic_cube[result_index, layer, head]:.3e}",
                flush=True,
            )
    lower, upper = publication_log_limits(
        structural_cube[keep],
        semantic_cube[keep],
    )

    for result_index, result in enumerate(results):
        semantic = np.asarray(result["semantic_score"], dtype=float)
        structural = np.asarray(result["structural_score"], dtype=float)
        selected_semantic = set(map(tuple, result["selected_groups"]["semantic"]))
        selected_structural = set(map(tuple, result["selected_groups"]["structural"]))
        marker = SEED_MARKERS[result_index % len(SEED_MARKERS)]

        for layer in range(cfg.layers):
            for head in range(cfg.heads):
                if not bool(keep[result_index, layer, head]):
                    continue
                x_value = float(structural[layer, head])
                y_value = float(semantic[layer, head])
                ax.scatter(
                    x_value,
                    y_value,
                    s=27.0,
                    marker=marker,
                    facecolor=colors[layer],
                    edgecolor=PAPER_COLOR,
                    linewidth=0.45,
                    alpha=0.90,
                    zorder=3,
                )
                ring_color = (
                    SEMANTIC_COLOR
                    if (layer, head) in selected_semantic
                    else STRUCTURAL_COLOR
                    if (layer, head) in selected_structural
                    else None
                )
                if ring_color is not None:
                    ax.scatter(
                        x_value,
                        y_value,
                        s=50.0,
                        marker=marker,
                        facecolor="none",
                        edgecolor=PAPER_COLOR,
                        linewidth=2.5,
                        zorder=4,
                    )
                    ax.scatter(
                        x_value,
                        y_value,
                        s=50.0,
                        marker=marker,
                        facecolor="none",
                        edgecolor=ring_color,
                        linewidth=1.35,
                        zorder=5,
                    )

    ax.plot(
        [lower, upper],
        [lower, upper],
        color=REFERENCE_COLOR,
        linewidth=0.78,
        linestyle=(0, (3.0, 2.2)),
        zorder=1,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_xlabel(r"Structural score $S_{\mathrm{str}}$")
    ax.set_ylabel(r"Semantic score $S_{\mathrm{sem}}$")
    ax.set_box_aspect(1.0)
    ax.set_title("(a)  Raw task scores", loc="left")
    style_scatter_axes(ax, xgrid=True, ygrid=True)

    handles, labels = selection_legend_handles()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.045),
        ncol=2,
        columnspacing=0.80,
        handletextpad=0.30,
        borderaxespad=0,
        fontsize=7.0,
    )
    fig.subplots_adjust(left=0.19, right=0.975, bottom=0.22, top=0.88)
    return save_figure(fig, figures_dir / "fig1a_raw_task_scores")


'''


DJ_SCATTER = r'''def _dj_scatter(
    ax: Any,
    rows: Sequence[Mapping[str, Any]],
    cfg: Config,
    x_key: str,
    y_key: str,
    *,
    reliable_only: bool = False,
    dim_unreliable: bool = False,
    marker_size: float = 27.0,
) -> None:
    """Draw every head with layer colour and seed shape at equal prominence."""

    colors = layer_palette(cfg.layers)
    seeds = sorted({int(row["seed"]) for row in rows})
    seed_indices = {seed: index for index, seed in enumerate(seeds)}
    for row in rows:
        reliable = bool(row["selectivity_reliable"])
        if reliable_only and not reliable:
            continue
        seed_index = seed_indices[int(row["seed"])]
        ax.scatter(
            float(row[x_key]),
            float(row[y_key]),
            s=marker_size,
            marker=SEED_MARKERS[seed_index % len(SEED_MARKERS)],
            facecolor=colors[int(row["layer"])],
            edgecolor=PAPER_COLOR,
            linewidth=0.45,
            alpha=0.90,
            zorder=3,
        )


'''


JOINT_SELECTIVITY_FIGURE = r'''def figure_joint_selectivity_plane(
    rows: Sequence[Mapping[str, Any]],
    cfg: Config,
    figures_dir: Path,
) -> list[str]:
    """Sensitivity/selectivity on the matched right-hand half-width canvas."""

    plt = configure_matplotlib()
    fig, ax = plt.subplots(figsize=PAIR_FIGSIZE)
    seeds = sorted({int(row["seed"]) for row in rows})

    ax.axvspan(
        -DJ_SELECTIVITY_THRESHOLD,
        DJ_SELECTIVITY_THRESHOLD,
        color=NEUTRAL_BAND_COLOR,
        linewidth=0,
        zorder=0,
    )
    ax.axvline(-DJ_SELECTIVITY_THRESHOLD, color=GRID_COLOR, linewidth=0.52, zorder=1)
    ax.axvline(DJ_SELECTIVITY_THRESHOLD, color=GRID_COLOR, linewidth=0.52, zorder=1)
    ax.axvline(0.0, color=REFERENCE_COLOR, linewidth=0.66, zorder=1)
    _dj_scatter(
        ax,
        rows,
        cfg,
        "selectivity_D",
        "joint_score_J",
        dim_unreliable=False,
    )

    for x_position, label, color in (
        (0.18, "structural", STRUCTURAL_COLOR),
        (0.50, "generalist", MUTED_COLOR),
        (0.82, "semantic", SEMANTIC_COLOR),
    ):
        ax.text(
            x_position,
            0.965,
            label,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=7.0,
            color=color,
            bbox={
                "facecolor": PAPER_COLOR,
                "edgecolor": "none",
                "alpha": 0.78,
                "pad": 0.4,
            },
            zorder=6,
        )
    ax.set_xlim(-1.02, 1.02)
    ax.set_yscale("log")
    ax.set_xlabel(r"Head selectivity $D_{\mathrm{rel}}$")
    ax.set_ylabel(r"Joint sensitivity $J$")
    ax.set_box_aspect(1.0)
    ax.set_title("(b)  Sensitivity and selectivity", loc="left")
    style_scatter_axes(ax, xgrid=False, ygrid=True)
    add_head_legend(fig, cfg, seeds, y=0.045)
    fig.subplots_adjust(left=0.19, right=0.975, bottom=0.22, top=0.88)
    return save_figure(fig, figures_dir / "fig1b_sensitivity_selectivity")


'''


CAUSAL_FIGURES = r'''def _rho_note(ax: Any, text: str) -> None:
    ax.text(
        0.045,
        0.955,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
        color=INK_COLOR,
        linespacing=1.22,
        bbox={
            "boxstyle": "round,pad=0.24",
            "facecolor": PAPER_COLOR,
            "edgecolor": "#D7DCE1",
            "linewidth": 0.55,
            "alpha": 0.96,
        },
        zorder=10,
    )


def figure_joint_selectivity_validation(
    rows: Sequence[Mapping[str, Any]],
    cfg: Config,
    figures_dir: Path,
    *,
    show_all_head_rho: bool = True,
) -> tuple[list[str], dict[str, Any], list[dict[str, Any]]]:
    """Render the causal-validation triptych as three matched component files."""

    plt = configure_matplotlib()
    summary: dict[str, Any] = {
        "score_calibration": "each channel divided by its within-seed head mean",
        "J_formula": "0.5 * (semantic_score_norm + structural_score_norm)",
        "D_formula": (
            "(semantic_score_norm - structural_score_norm) / "
            "(semantic_score_norm + structural_score_norm)"
        ),
        "J_reliability_floor": DJ_RELIABILITY_FLOOR,
        "D_specialist_threshold": DJ_SELECTIVITY_THRESHOLD,
        # Both statistics are an approved, unconditional part of panels b and c.
        "show_all_head_rho": True,
    }

    joint_stats = correlation_by_seed(rows, "joint_score_J", "ablation_joint_impact")
    role_reliable = correlation_by_seed(
        rows,
        "selectivity_D",
        "ablation_role_selectivity",
        reliable_only=True,
    )
    role_all = correlation_by_seed(
        rows,
        "selectivity_D",
        "ablation_role_selectivity",
        reliable_only=False,
    )
    rescue_reliable = correlation_by_seed(
        rows,
        "selectivity_D",
        "rescue_role_contrast",
        reliable_only=True,
    )
    rescue_all = correlation_by_seed(
        rows,
        "selectivity_D",
        "rescue_role_contrast",
        reliable_only=False,
    )
    summary["J_vs_joint_ablation"] = joint_stats
    summary["D_vs_ablation_role"] = role_reliable
    summary["D_vs_ablation_role_all_heads"] = role_all
    summary["D_vs_rescue_role"] = rescue_reliable
    summary["D_vs_rescue_role_all_heads"] = rescue_all

    specifications = (
        {
            "x_key": "joint_score_J",
            "y_key": "ablation_joint_impact",
            "xlabel": r"Joint sensitivity $J$",
            "ylabel": "Ablation impact",
            "title": "(a) Sensitivity vs impact",
            "stem": "fig2a_sensitivity_impact",
            "xlog": True,
            "ylog": True,
            "annotation": rf"All heads: $\rho = {joint_stats['pooled_spearman']:.2f}$",
        },
        {
            "x_key": "selectivity_D",
            "y_key": "ablation_role_selectivity",
            "xlabel": r"Selectivity $D_{\mathrm{rel}}$",
            "ylabel": "Ablation contrast",
            "title": "(b) Selectivity vs ablation",
            "stem": "fig2b_selectivity_ablation",
            "fixed_unit_range": True,
            "annotation": "\n".join((
                rf"All heads: $\rho = {role_all['pooled_spearman']:.2f}$",
                rf"$J \geq 0.5$: $\rho = {role_reliable['pooled_spearman']:.2f}$",
            )),
        },
        {
            "x_key": "selectivity_D",
            "y_key": "rescue_role_contrast",
            "xlabel": r"Selectivity $D_{\mathrm{rel}}$",
            "ylabel": "Rescue contrast",
            "title": "(c) Selectivity vs rescue",
            "stem": "fig2c_selectivity_rescue",
            "annotation": "\n".join((
                rf"All heads: $\rho = {rescue_all['pooled_spearman']:.2f}$",
                rf"$J \geq 0.5$: $\rho = {rescue_reliable['pooled_spearman']:.2f}$",
            )),
        },
    )

    paths: list[str] = []
    for index, specification in enumerate(specifications):
        fig, ax = plt.subplots(figsize=CAUSAL_FIGSIZE)
        _dj_scatter(
            ax,
            rows,
            cfg,
            specification["x_key"],
            specification["y_key"],
            dim_unreliable=False,
            marker_size=23.5,
        )
        if specification.get("xlog"):
            ax.set_xscale("log")
        if specification.get("ylog"):
            ax.set_yscale("log")
        if index > 0:
            ax.axhline(0.0, color=REFERENCE_COLOR, linewidth=0.62, zorder=1)
            ax.axvline(0.0, color=REFERENCE_COLOR, linewidth=0.62, zorder=1)
            ax.set_xlim(-1.03, 1.03)
        if specification.get("fixed_unit_range"):
            ax.set_ylim(-1.03, 1.03)
        ax.set_title(
            specification["title"],
            loc="left",
            fontsize=7.7,
            fontweight="semibold",
            pad=4.0,
        )
        ax.set_xlabel(specification["xlabel"], fontsize=8.0)
        ax.set_ylabel(specification["ylabel"], fontsize=8.0)
        _rho_note(ax, specification["annotation"])
        style_scatter_axes(ax, xgrid=False, ygrid=True)
        # Symmetric safety margins keep scientific-notation ticks inside each
        # fixed third-width page when the three PDFs are placed side by side.
        fig.subplots_adjust(left=0.275, right=0.925, bottom=0.22, top=0.84)
        paths.extend(save_figure(fig, figures_dir / specification["stem"]))

    quadrant_rows = dj_quadrant_rows(rows)
    summary["quadrants_by_seed"] = quadrant_rows
    return paths, summary, quadrant_rows


'''


MATRIX_FIGURES = r'''def _clean_symmetric_limit(matrix: np.ndarray) -> float:
    finite = np.abs(np.asarray(matrix, dtype=float))
    finite = finite[np.isfinite(finite)]
    maximum = float(finite.max()) if finite.size else 1.0
    maximum = max(maximum, 1.0e-12)
    exponent = math.floor(math.log10(maximum))
    scale = 10.0 ** exponent
    normalized = maximum / scale
    choices = (1.0, 1.2, 1.5, 1.6, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0)
    rounded = next(choice for choice in choices if choice >= normalized - 1.0e-12)
    return rounded * scale


def _draw_publication_matrix(
    matrix: np.ndarray,
    figures_dir: Path,
    *,
    title: str,
    stem: str,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    xlabel: str,
    ylabel: str,
    colorbar_label: str,
) -> list[str]:
    import matplotlib as mpl
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    from matplotlib.patches import Rectangle

    plt = configure_matplotlib()
    matrix = np.asarray(matrix, dtype=float)
    fig = plt.figure(figsize=MATRIX_FIGSIZE)
    ax = fig.add_axes([0.27, 0.305, 0.63, 0.60])
    colorbar_ax = fig.add_axes([0.27, 0.105, 0.66, 0.034])
    limit = _clean_symmetric_limit(matrix)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    cmap = LinearSegmentedColormap.from_list(
        "effect_blue_vermillion",
        EFFECT_CMAP_COLORS,
    )

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = float(matrix[row, col])
            color = cmap(norm(value))
            ax.add_patch(Rectangle(
                (col - 0.5, row - 0.5),
                1.0,
                1.0,
                facecolor=color,
                edgecolor=PAPER_COLOR,
                linewidth=1.15,
            ))
            red, green, blue, _ = color
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            ax.text(
                col,
                row,
                f"{value:+.3f}",
                ha="center",
                va="center",
                fontsize=8.3,
                fontweight="semibold",
                color=INK_COLOR if luminance > 0.57 else PAPER_COLOR,
            )

    ax.set_xlim(-0.5, matrix.shape[1] - 0.5)
    ax.set_ylim(matrix.shape[0] - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(len(col_labels)), col_labels)
    ax.set_yticks(np.arange(len(row_labels)), row_labels)
    ax.tick_params(axis="both", length=0, pad=4, labelsize=7.0)
    ax.set_xlabel(xlabel, fontsize=8.2, labelpad=1.0)
    ax.set_ylabel(ylabel, fontsize=8.2)
    fig.text(
        0.27,
        0.965,
        title,
        ha="left",
        va="top",
        fontsize=8.6,
        fontweight="semibold",
    )
    for spine in ax.spines.values():
        spine.set_visible(False)

    colorbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=colorbar_ax,
        orientation="horizontal",
        ticks=(-limit, 0.0, limit),
    )
    decimals = max(0, -math.floor(math.log10(limit)) + 1)
    colorbar.ax.set_xticklabels([
        f"{-limit:.{decimals}f}",
        f"{0.0:.{decimals}f}",
        f"{limit:.{decimals}f}",
    ])
    colorbar.set_label(colorbar_label, labelpad=1.0, fontsize=7.0)
    colorbar.outline.set_linewidth(0.5)
    colorbar.outline.set_edgecolor(AXIS_COLOR)
    colorbar.ax.tick_params(labelsize=7.0, length=2.0, pad=1.3)
    return save_figure(fig, figures_dir / stem)


def figure_necessity_matrix(
    values: Mapping[str, Any],
    figures_dir: Path,
) -> list[str]:
    return _draw_publication_matrix(
        np.asarray(values["necessity_mean"], dtype=float),
        figures_dir,
        title="(a) Necessity",
        stem="fig4a_necessity",
        row_labels=("Semantic", "Structural"),
        col_labels=("Semantic", "Structural"),
        xlabel="Evaluation task",
        ylabel="Ablated family",
        colorbar_label="Δ Cross-Entropy",
    )


'''


RESCUE_MATRIX_FIGURE = r'''def figure_rescue_matrix(
    values: Mapping[str, Any],
    figures_dir: Path,
) -> list[str]:
    return _draw_publication_matrix(
        np.asarray(values["rescue_mean"], dtype=float),
        figures_dir,
        title="(b) Rescue",
        stem="fig4b_rescue",
        row_labels=("Semantic", "Structural"),
        col_labels=("Semantic", "Structural"),
        xlabel="Corruption",
        ylabel="Patched family",
        colorbar_label="Mediated fraction",
    )


'''


ATTENTION_FIGURE = r'''def figure_attention_visualisations(
    payload: Mapping[str, Any],
    cfg: Config,
    figures_dir: Path,
) -> list[str]:
    """Four coherent head panels: query graph plus vector attention matrix."""

    import matplotlib as mpl
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    plt = configure_matplotlib()
    examples = list(payload["examples"])
    if len(examples) != 4:
        raise ValueError("attention publication figure requires exactly four examples")
    matrices = [
        np.asarray(example["attention_matrix"], dtype=float)
        for example in examples
    ]
    if any(matrix.shape != (cfg.n, cfg.n) for matrix in matrices):
        raise ValueError("attention matrices must match the configured graph size")

    maximum = max(float(np.nanmax(matrix)) for matrix in matrices)
    norm = mpl.colors.Normalize(vmin=0.0, vmax=max(maximum, 1.0e-6))
    cmap = mpl.colormaps["cividis"]
    fig = plt.figure(figsize=ATTENTION_FIGSIZE)
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
    angles = np.linspace(
        np.pi / 2.0,
        np.pi / 2.0 + 2.0 * np.pi,
        cfg.n,
        endpoint=False,
    )
    positions = np.column_stack([np.cos(angles), np.sin(angles)])
    tick_nodes = np.unique(np.asarray([0, (cfg.n - 1) // 2, cfg.n - 1], dtype=int))

    for index, (example, matrix) in enumerate(zip(examples, matrices)):
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
        query = int(example["query_node"])
        source = int(example["source_node"])

        # Captured GRIT tensors are [source, destination].  The publication view
        # is destination/query rows by source/key columns, without renormalising.
        display_matrix = matrix.T
        query_attention = matrix[:, query]

        title_ax.axis("off")
        title_ax.text(
            0.0,
            0.52,
            f"({chr(97 + index)})  {example['role']}",
            ha="left",
            va="center",
            fontsize=8.4,
            fontweight="semibold",
        )
        title_ax.text(
            1.0,
            0.52,
            f"L{int(example['layer_display'])} H{int(example['head_display'])}",
            ha="right",
            va="center",
            fontsize=7.0,
            color=MUTED_COLOR,
        )

        for node in range(cfg.n):
            neighbour = (node + 1) % cfg.n
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
            s=57.0,
            edgecolors="#69727C",
            linewidths=0.45,
            zorder=3,
        )
        for selected, color, marker in (
            (query, QUERY_COLOR, "o"),
            (source, SOURCE_COLOR, "s"),
        ):
            graph_ax.scatter(
                [positions[selected, 0]],
                [positions[selected, 1]],
                s=105.0,
                marker=marker,
                facecolors="none",
                edgecolors=PAPER_COLOR,
                linewidths=2.0,
                zorder=5,
            )
            graph_ax.scatter(
                [positions[selected, 0]],
                [positions[selected, 1]],
                s=105.0,
                marker=marker,
                facecolors="none",
                edgecolors=color,
                linewidths=1.15,
                zorder=6,
            )
        for node, (x_position, y_position) in enumerate(positions):
            if node not in {query, source}:
                continue
            red, green, blue, _ = cmap(norm(query_attention[node]))
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            graph_ax.text(
                x_position,
                y_position,
                str(node + 1),
                ha="center",
                va="center",
                fontsize=6.4,
                fontweight="semibold",
                color=INK_COLOR if luminance > 0.57 else PAPER_COLOR,
                zorder=7,
            )
        graph_ax.set_xlim(-1.20, 1.20)
        graph_ax.set_ylim(-1.18, 1.18)
        graph_ax.set_aspect("equal")
        graph_ax.axis("off")

        cell_edges = np.arange(cfg.n + 1, dtype=float) - 0.5
        matrix_ax.pcolormesh(
            cell_edges,
            cell_edges,
            display_matrix,
            cmap=cmap,
            norm=norm,
            shading="flat",
            edgecolors="face",
            linewidth=0.15,
            antialiased=False,
            rasterized=False,
            snap=True,
        )
        matrix_ax.set_xlim(-0.5, cfg.n - 0.5)
        matrix_ax.set_ylim(cfg.n - 0.5, -0.5)
        matrix_ax.set_aspect("equal")
        for xy, width, height, color, linestyle in (
            ((-0.5, query - 0.5), cfg.n, 1.0, QUERY_COLOR, "-"),
            ((source - 0.5, -0.5), 1.0, cfg.n, SOURCE_COLOR, (0, (3.0, 1.8))),
        ):
            matrix_ax.add_patch(Rectangle(
                xy,
                width,
                height,
                fill=False,
                edgecolor=PAPER_COLOR,
                linewidth=1.35,
            ))
            matrix_ax.add_patch(Rectangle(
                xy,
                width,
                height,
                fill=False,
                edgecolor=color,
                linewidth=0.72,
                linestyle=linestyle,
            ))
        matrix_ax.set_xticks(tick_nodes, [str(node + 1) for node in tick_nodes])
        matrix_ax.set_yticks(tick_nodes, [str(node + 1) for node in tick_nodes])
        matrix_ax.tick_params(axis="both", labelsize=7.0, length=1.8, pad=1.2)
        matrix_ax.set_xlabel(
            "Source node" if index >= 2 else "",
            fontsize=7.0,
            labelpad=1.5,
        )
        matrix_ax.set_ylabel("")
        for spine in matrix_ax.spines.values():
            spine.set_visible(True)
            spine.set_color(AXIS_COLOR)
            spine.set_linewidth(0.55)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="none",
            markeredgecolor=QUERY_COLOR,
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
            markeredgecolor=SOURCE_COLOR,
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
    colorbar.ax.tick_params(labelsize=7.0, length=1.8, pad=1.1)
    colorbar.ax.set_title(
        "Attention weight",
        fontsize=7.0,
        color=MUTED_COLOR,
        pad=2.5,
    )
    colorbar.outline.set_linewidth(0.5)
    return save_figure(fig, figures_dir / "fig5_attention_visualisations")


'''


ABLATION_FIGURE = r'''def figure_dj_family_ablation(
    results: Sequence[dict[str, Any]],
    cfg: Config,
    figures_dir: Path,
) -> tuple[list[str], dict[str, Any]]:
    """Joint ablation grid with across-seed mean and observed seed range."""

    plt = configure_matplotlib()
    values = dj_family_ablation_values(results)
    fig, axes = plt.subplots(
        2,
        2,
        figsize=ABLATION_FIGSIZE,
        sharex=True,
        sharey="row",
    )
    styles = {
        "semantic_specialist": (SEMANTIC_COLOR, "o", "-", "Semantic"),
        "structural_specialist": (
            STRUCTURAL_COLOR,
            "s",
            (0, (4.0, 1.8)),
            "Structural",
        ),
        "high_J_generalist": (
            GENERALIST_COLOR,
            "D",
            (0, (2.3, 1.4)),
            "Generalist",
        ),
        "low_J_inert": (INERT_COLOR, "^", (0, (1.0, 1.6)), "Inert"),
    }
    metrics = (
        ("functional_by_seed", "Functional logit impact", 1.0),
        ("accuracy_drop_by_seed", "Accuracy drop (pp)", 100.0),
    )
    maximum_prefix = 0

    for column, task_name in enumerate(("semantic", "structural")):
        for row_index, (metric, ylabel, scale) in enumerate(metrics):
            ax = axes[row_index, column]
            for family_name, (color, marker, linestyle, label) in styles.items():
                curves = np.asarray(
                    values[task_name][family_name][metric],
                    dtype=float,
                ) * scale
                x_values = np.arange(curves.shape[1])
                maximum_prefix = max(maximum_prefix, int(x_values[-1]))
                mean = np.nanmean(curves, axis=0)
                lower = np.nanmin(curves, axis=0)
                upper = np.nanmax(curves, axis=0)
                ax.fill_between(
                    x_values,
                    lower,
                    upper,
                    color=color,
                    alpha=0.075,
                    linewidth=0,
                )
                ax.plot(
                    x_values,
                    mean,
                    color=color,
                    marker=marker,
                    linestyle=linestyle,
                    linewidth=1.65,
                    markersize=3.4,
                    markeredgecolor=PAPER_COLOR,
                    markeredgewidth=0.35,
                    label=label,
                    zorder=3,
                )
            ax.axhline(0.0, color=REFERENCE_COLOR, linewidth=0.66, zorder=1)
            if column == 0:
                ax.set_ylabel(ylabel)
            if row_index == 0:
                ax.set_title(f"{task_name.capitalize()} task")
            panel_index = row_index * 2 + column
            ax.text(
                -0.02,
                1.055,
                f"({chr(97 + panel_index)})",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontweight="semibold",
                fontsize=8.7,
                clip_on=False,
            )
            style_scatter_axes(ax, xgrid=False, ygrid=True)

    for ax in axes.ravel():
        ax.set_xticks(np.arange(maximum_prefix + 1))
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
    fig.subplots_adjust(
        left=0.105,
        right=0.982,
        bottom=0.17,
        top=0.90,
        hspace=0.25,
        wspace=0.22,
    )
    return save_figure(fig, figures_dir / "fig3_joint_ablation_tests"), values


VALIDATION_METRIC_LABELS = (
    ("accuracy", "overall accuracy"),
    ("semantic_accuracy", "semantic accuracy"),
    ("structural_accuracy", "structural accuracy"),
    ("loss", "overall loss"),
    ("semantic_loss", "semantic loss"),
    ("structural_loss", "structural loss"),
)


'''


CREATE_OUTPUTS = r'''def create_outputs(
    results: Sequence[dict[str, Any]],
    cfg: Config,
    run_dir: Path,
    validation: Mapping[str, Any] | None = None,
    *,
    show_all_head_rho: bool = True,
    attention_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    figures_dir = run_dir / "figures"
    tables_dir = run_dir / "tables"
    head_rows = build_head_rows(results)
    write_csv(tables_dir / "per_head_metrics.csv", head_rows)

    # Preserve every machine-readable analysis while rendering nine matched
    # publication assets designed for their final side-by-side paper placement.
    correlations = score_ablation_correlations(head_rows)
    iterative_ablation = iterative_family_ablation_values(results)
    dissociation = double_dissociation_values(results)

    fig1a = figure_specialisation_plane(results, cfg, figures_dir)
    fig1b = figure_joint_selectivity_plane(head_rows, cfg, figures_dir)
    fig2, joint_selectivity_summary, quadrant_rows = (
        figure_joint_selectivity_validation(
            head_rows,
            cfg,
            figures_dir,
            show_all_head_rho=show_all_head_rho,
        )
    )
    fig3, dj_family_ablation = figure_dj_family_ablation(results, cfg, figures_dir)
    fig4a = figure_necessity_matrix(dissociation, figures_dir)
    fig4b = figure_rescue_matrix(dissociation, figures_dir)
    fig5 = (
        figure_attention_visualisations(attention_payload, cfg, figures_dir)
        if attention_payload is not None
        else []
    )
    write_csv(tables_dir / "joint_selectivity_quadrants.csv", quadrant_rows)
    attention_summary = (
        attention_visualisation_summary(attention_payload)
        if attention_payload is not None
        else None
    )
    if attention_summary is not None:
        write_json(
            tables_dir / "attention_visualisation_heads.json",
            attention_summary,
        )

    seed_summaries = []
    for result in results:
        seed_summaries.append({
            "seed": int(result["seed"]),
            "selected_groups": result["selected_groups"],
            "checks": result["checks"],
            "semantic_clean_accuracy": result["ablation_semantic"]["clean_accuracy"],
            "structural_clean_accuracy": result["ablation_structural"]["clean_accuracy"],
            "semantic_corrupt_clean_label_accuracy": result["rescue_semantic"]["corrupt_clean_label_accuracy"],
            "structural_corrupt_clean_label_accuracy": result["rescue_structural"]["corrupt_clean_label_accuracy"],
            "semantic_corruption_effect_norm": finite_mean(result["rescue_semantic"]["effect_norm"]),
            "structural_corruption_effect_norm": finite_mean(result["rescue_structural"]["effect_norm"]),
        })
    summary = {
        "version": EXPERIMENT_VERSION,
        "analysis_version": ANALYSIS_VERSION,
        "figure_version": FIGURE_VERSION,
        "fingerprint": config_fingerprint(cfg),
        "analysis_fingerprint": analysis_fingerprint(cfg),
        "official_grit_commit": OFFICIAL_GRIT_COMMIT,
        "config": asdict(cfg),
        "figure_options": {
            "show_all_head_rho": True,
            "attention_visualisation": attention_payload is not None,
            "attention_colour_normalization": (
                "shared linear Normalize(vmin=0, vmax=global maximum)"
            ),
            "attention_colour_palette": "Matplotlib cividis",
            "attention_matrix_display": (
                "vector transpose of captured [source, destination] tensor; "
                "rows are destination/query and sum to one"
            ),
            "attention_node_marker_area_points_squared": 57.0,
            "figure_asset_count": 9 if attention_payload is not None else 8,
            "png_dpi": PNG_DPI,
            "pdf_raster_dpi": PDF_RASTER_DPI,
            "pdf_vector_output": True,
            "attention_matrices_vector": True,
            "joint_ablation_band": "observed across-seed range",
            "inline_preview_width_pixels": {
                "component": INLINE_COMPONENT_WIDTH,
                "full_width": INLINE_FULL_WIDTH,
            },
            "raw_score_outlier_policy": (
                "display-only omission of every head whose semantic or structural "
                f"score is below {RAW_SCORE_DISPLAY_FLOOR:.0e}"
            ),
        },
        "score_ablation_correlations": correlations,
        "necessity_mean": dissociation["necessity_mean"],
        "rescue_mean": dissociation["rescue_mean"],
        "necessity_interaction_by_seed": dissociation["necessity_interaction"],
        "rescue_interaction_by_seed": dissociation["rescue_interaction"],
        "selected_heads": dissociation["selected_heads"],
        "iterative_family_ablation": iterative_ablation,
        "joint_influence_selectivity": joint_selectivity_summary,
        "joint_selectivity_family_ablation": dj_family_ablation,
        "attention_visualisation": attention_summary,
        "validation_performance": validation,
        "seeds": seed_summaries,
        "figures": fig1a + fig1b + fig2 + fig3 + fig4a + fig4b + fig5,
    }
    write_json(run_dir / "summary.json", summary)
    write_json(
        tables_dir / "selected_heads.json",
        {"selected_heads": dissociation["selected_heads"]},
    )
    return summary


'''


OUTPUT_DOCUMENTATION = r'''Primary outputs (PNG + vector PDF)
----------------------------------
``fig1a_raw_task_scores`` and ``fig1b_sensitivity_selectivity``
    Matched half-width panels for raw structural-versus-semantic scores and the
    calibrated joint-sensitivity/selectivity (J--D_rel) head plane. The raw-score
    view omits heads whose semantic or structural score is below 1e-4; the
    underlying table and every downstream analysis remain unchanged.
``fig2a_sensitivity_impact``, ``fig2b_selectivity_ablation``, and
``fig2c_selectivity_rescue``
    Three matched third-width causal-validation panels, designed to sit side by side.
    Panels b and c report both all-head and reliable-head (J >= 0.5) correlations.
``fig3_joint_ablation_tests``
    Full-width cumulative ablations of semantic specialists, structural specialists,
    high-J generalists, and low-J/inert heads on both tasks. Bands show the observed
    across-seed range.
``fig4a_necessity`` and ``fig4b_rescue``
    Matched half-width double-dissociation matrices, designed to sit side by side.
``fig5_attention_visualisations``
    Full-width attention-weighted input graphs and vector attention matrices for the
    strongest reliable semantic specialist, structural specialist, high-J generalist,
    and a first-layer head across trained seeds.
'''


def replace_between(source: str, start: str, end: str, replacement: str) -> str:
    start_index = source.index(start)
    end_index = source.index(end, start_index)
    return source[:start_index] + replacement + source[end_index:]


def replace_function_block(
    source: str,
    start_name: str,
    next_name: str,
    replacement: str,
) -> str:
    return replace_between(
        source,
        f"def {start_name}(",
        f"def {next_name}(",
        replacement,
    )


def transform(source: str) -> str:
    source = replace_between(
        source,
        'FIGURE_VERSION = "mixed-task-publication-classic-v8"',
        "\n\n# ======================================================================================\n# Colab/bootstrap helpers",
        PUBLICATION_CONSTANTS,
    )
    source = replace_between(
        source,
        "Primary outputs (PNG + vector PDF)\n----------------------------------\n",
        "``tables/validation_performance.csv``",
        OUTPUT_DOCUMENTATION,
    )
    source = replace_function_block(
        source,
        "configure_matplotlib",
        "robust_shared_log_limits",
        PUBLICATION_HELPERS,
    )
    source = replace_function_block(
        source,
        "figure_specialisation_plane",
        "score_ablation_correlations",
        RAW_SCORE_FIGURE,
    )
    source = replace_function_block(
        source,
        "_dj_scatter",
        "figure_joint_selectivity_plane",
        DJ_SCATTER,
    )
    source = replace_function_block(
        source,
        "figure_joint_selectivity_plane",
        "_rho_annotation",
        JOINT_SELECTIVITY_FIGURE,
    )
    source = replace_function_block(
        source,
        "_rho_annotation",
        "draw_heatmap",
        CAUSAL_FIGURES,
    )
    source = replace_function_block(
        source,
        "figure_necessity_matrix",
        "figure_rescue_matrix",
        MATRIX_FIGURES,
    )
    source = replace_function_block(
        source,
        "figure_rescue_matrix",
        "attention_visualisation_summary",
        RESCUE_MATRIX_FIGURE,
    )
    source = replace_function_block(
        source,
        "figure_attention_visualisations",
        "iterative_family_ablation_values",
        ATTENTION_FIGURE,
    )
    source = replace_function_block(
        source,
        "figure_dj_family_ablation",
        "validation_performance_summary",
        ABLATION_FIGURE,
    )
    source = replace_function_block(
        source,
        "create_outputs",
        "environment_record",
        CREATE_OUTPUTS,
    )
    return source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    notebook = json.loads(args.source.read_text(encoding="utf-8"))
    populated_cells = [
        cell
        for cell in notebook.get("cells", [])
        if "".join(cell.get("source", [])).strip()
    ]
    if len(populated_cells) != 1 or populated_cells[0].get("cell_type") != "code":
        raise ValueError("expected exactly one populated code cell")

    source = "".join(populated_cells[0]["source"])
    transformed = transform(source)
    compile(transformed, str(args.output), "exec")

    cell = dict(populated_cells[0])
    cell["source"] = transformed.splitlines(keepends=True)
    cell["outputs"] = []
    cell["execution_count"] = None
    notebook["cells"] = [cell]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
