"""One publication figure system for every canonical task."""

from __future__ import annotations

import dataclasses
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .cache import atomic_json
from .scores import (
    JOINT_AXIS_LABEL,
    SELECTIVITY_AXIS_LABEL,
    SEMANTIC_AXIS_LABEL,
    STRUCTURAL_AXIS_LABEL,
    HeadCoordinates,
)


TASK_FIGURE_MODIFIERS: dict[str, Callable[[str, Any, Any], None]] = {}


def register_task_figure_modifier(
    task: str, modifier: Callable[[str, Any, Any], None]
) -> None:
    """Register a presentation-only task hook; estimators and cached values stay untouched."""

    TASK_FIGURE_MODIFIERS[str(task)] = modifier


@dataclass(frozen=True)
class FigureTheme:
    width: float = 5.0
    height: float = 4.4
    dpi: int = 350
    font_family: str = "DejaVu Sans"
    font_size: float = 9.5
    label_size: float = 10.5
    title_size: float = 11.0
    tick_size: float = 8.5
    line_width: float = 1.3
    marker_size: float = 34.0
    error_alpha: float = 0.22
    grid_alpha: float = 0.18
    layer_cmap: str = "viridis"
    # Presentation-only cap on distance columns per figure; long-diameter tasks are grouped to fit.
    max_distance_points: int = 14
    semantic_color: str = "#C44E52"
    structural_color: str = "#4C72B0"
    central_color: str = "#6E6E6E"
    functional_color: str = "#4C72B0"
    beneficial_color: str = "#55A868"
    inactive_color: str = "#B8B8B8"
    formats: tuple[str, ...] = ("pdf", "png")

    def with_overrides(self, values: Mapping[str, Any] | None) -> "FigureTheme":
        if not values:
            return self
        unknown = sorted(set(values) - set(self.__dataclass_fields__))
        if unknown:
            raise ValueError(f"unknown figure theme settings: {unknown}")
        return replace(self, **dict(values))


@contextmanager
def publication_style(theme: FigureTheme):
    import matplotlib as mpl

    settings = {
        "font.family": theme.font_family,
        "font.size": theme.font_size,
        "axes.labelsize": theme.label_size,
        "axes.titlesize": theme.title_size,
        "xtick.labelsize": theme.tick_size,
        "ytick.labelsize": theme.tick_size,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
    }
    with mpl.rc_context(settings):
        yield


@dataclass(frozen=True)
class HeadPlotData:
    coordinates: HeadCoordinates
    seed: int
    semantic_interval: tuple[np.ndarray, np.ndarray] | None = None
    structural_interval: tuple[np.ndarray, np.ndarray] | None = None
    joint_interval: tuple[np.ndarray, np.ndarray] | None = None
    selectivity_interval: tuple[np.ndarray, np.ndarray] | None = None


def _flatten_heads(array: Any) -> np.ndarray:
    return np.asarray(array).reshape(-1)


def _exact_panel_title(grouped: bool) -> str:
    """Name the exact panel for what it plots; grouped columns show mass per unit distance."""

    return (
        "Exact score contribution per unit distance"
        if grouped
        else "Exact score contribution"
    )


def distance_ticks(ax, labels: Sequence[int | str], theme: FigureTheme) -> None:
    """Label every distance column, rotating once grouped ranges make the labels wide.

    Callers pass an already-grouped axis (see `distance.display_bins`), so this only has to keep
    the labels legible, never to thin them: a distance figure with hidden columns invites the
    reader to interpolate across a gap that may not be there.
    """

    text = [str(value) for value in labels]
    ax.set_xticks(np.arange(len(text)), text)
    if any(len(value) > 3 for value in text) or len(text) > 10:
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_horizontalalignment("right")
            label.set_rotation_mode("anchor")


def _head_colours(shape: tuple[int, int], cmap: str):
    import matplotlib.pyplot as plt

    layer = np.repeat(np.arange(shape[0]), shape[1])
    normalizer = plt.Normalize(0, max(1, shape[0] - 1))
    return plt.get_cmap(cmap)(normalizer(layer)), layer, normalizer


def _errorbars(ax, x, y, x_interval, y_interval, theme):
    if x_interval is None or y_interval is None:
        return
    xlo, xhi = (_flatten_heads(value) for value in x_interval)
    ylo, yhi = (_flatten_heads(value) for value in y_interval)
    ax.errorbar(
        x,
        y,
        xerr=np.vstack((x - xlo, xhi - x)),
        yerr=np.vstack((y - ylo, yhi - y)),
        fmt="none",
        ecolor="#555555",
        alpha=theme.error_alpha,
        linewidth=0.55,
        capsize=0,
        zorder=1,
    )


def score_plane(
    data: HeadPlotData,
    *,
    title: str | None = None,
    theme: FigureTheme = FigureTheme(),
):
    import matplotlib.pyplot as plt

    coordinates = data.coordinates
    x = _flatten_heads(coordinates.normalized_structural)
    y = _flatten_heads(coordinates.normalized_semantic)
    colours, layers, normalizer = _head_colours(
        coordinates.raw_semantic.shape, theme.layer_cmap
    )
    with publication_style(theme):
        fig, ax = plt.subplots(figsize=(theme.width, theme.height))
        _errorbars(
            ax,
            x,
            y,
            data.structural_interval,
            data.semantic_interval,
            theme,
        )
        ax.scatter(
            x,
            y,
            c=colours,
            s=theme.marker_size,
            edgecolor="white",
            linewidth=0.45,
            zorder=2,
        )
        finite = np.concatenate((x[np.isfinite(x)], y[np.isfinite(y)]))
        upper = max(1.05, float(np.max(finite)) * 1.08) if finite.size else 1.05
        ax.plot([0, upper], [0, upper], color="#777777", linestyle="--", linewidth=0.9)
        ax.set(xlim=(0, upper), ylim=(0, upper))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(STRUCTURAL_AXIS_LABEL)
        ax.set_ylabel(SEMANTIC_AXIS_LABEL)
        if title:
            ax.set_title(title)
        ax.grid(alpha=theme.grid_alpha, linewidth=0.5)
        scalar = plt.cm.ScalarMappable(norm=normalizer, cmap=theme.layer_cmap)
        bar = fig.colorbar(scalar, ax=ax, pad=0.025)
        bar.set_label("Layer")
        bar.set_ticks(np.arange(coordinates.raw_semantic.shape[0]))
    return fig, ax


def joint_selectivity_plane(
    data: HeadPlotData,
    *,
    title: str | None = None,
    theme: FigureTheme = FigureTheme(),
):
    import matplotlib.pyplot as plt

    coordinates = data.coordinates
    x = _flatten_heads(coordinates.selectivity)
    y = _flatten_heads(coordinates.joint_sensitivity)
    active = _flatten_heads(coordinates.active).astype(bool)
    colours, _, normalizer = _head_colours(
        coordinates.raw_semantic.shape, theme.layer_cmap
    )
    colours[~active] = np.asarray(
        __import__("matplotlib").colors.to_rgba(theme.inactive_color)
    )
    with publication_style(theme):
        fig, ax = plt.subplots(figsize=(theme.width, theme.height))
        _errorbars(ax, x, y, data.selectivity_interval, data.joint_interval, theme)
        ax.scatter(
            x,
            y,
            c=colours,
            s=theme.marker_size,
            edgecolor="white",
            linewidth=0.45,
        )
        ax.axvline(0, color="#777777", linestyle="--", linewidth=0.9)
        ax.axhline(
            float(np.nanmin(y[active])) if active.any() else 0,
            color="#999999",
            linestyle=":",
            linewidth=0.8,
        )
        ax.set_xlim(-1.03, 1.03)
        ax.set_xlabel(SELECTIVITY_AXIS_LABEL)
        ax.set_ylabel(JOINT_AXIS_LABEL)
        if title:
            ax.set_title(title)
        ax.grid(alpha=theme.grid_alpha, linewidth=0.5)
        scalar = plt.cm.ScalarMappable(norm=normalizer, cmap=theme.layer_cmap)
        bar = fig.colorbar(scalar, ax=ax, pad=0.025)
        bar.set_label("Layer")
        bar.set_ticks(np.arange(coordinates.raw_semantic.shape[0]))
    return fig, ax


def distance_heatmaps(
    exact: np.ndarray,
    per_opportunity: np.ndarray,
    labels: Sequence[int | str],
    *,
    channel: str,
    title: str | None = None,
    normalised: bool = False,
    grouped: bool = False,
    theme: FigureTheme = FigureTheme(),
):
    """Head-resolved distance atlas: one row per head, layer 0 at the top.

    Both panels take ``[layer, head, distance]``.  Heads stay separate because a layer's heads
    routinely peak at different distances, and summing them reports every layer as broader than any
    head inside it.  ``normalised`` only relabels a figure whose panels the caller already divided
    by each head's own total; the geometry and panel titles are otherwise identical.
    """

    import matplotlib.pyplot as plt

    exact = np.asarray(exact)
    per_opportunity = np.asarray(per_opportunity)
    if exact.shape != per_opportunity.shape:
        raise ValueError("exact/support-normalized heatmaps must have the same geometry")
    if exact.ndim != 3:
        raise ValueError(
            f"distance heatmaps need [layer,head,distance]; got {tuple(exact.shape)}"
        )
    layers, heads, distances = exact.shape
    if distances != len(labels):
        raise ValueError("heatmap distance axis does not match the registered labels")
    with publication_style(theme):
        fig, axes = plt.subplots(
            1, 2, figsize=(theme.width * 1.75, theme.height * 1.6), constrained_layout=True
        )
        for ax, matrix, panel in zip(
            axes,
            (exact, per_opportunity),
            (_exact_panel_title(grouped), "Contribution / event-carrier support"),
        ):
            # Row-major [L,H] with the default upper origin puts layer 0 in the top block.
            image = ax.imshow(
                matrix.reshape(layers * heads, distances),
                origin="upper",
                aspect="auto",
                interpolation="nearest",
                cmap="magma",
            )
            for layer in range(1, layers):
                ax.axhline(
                    layer * heads - 0.5, color="white", linewidth=0.35, alpha=0.7
                )
            ax.set_xlabel("Carrier distance from changed node")
            ax.set_ylabel("Head rows (layer blocks)")
            distance_ticks(ax, labels, theme)
            ax.set_yticks(
                [layer * heads + (heads - 1) / 2 for layer in range(layers)],
                [f"L{layer}" for layer in range(layers)],
            )
            ax.set_title(panel)
            fig.colorbar(image, ax=ax, pad=0.02)
        heading = title or f"{channel.capitalize()} donor-swap"
        fig.suptitle(f"{heading} (row-normalised)" if normalised else heading)
    return fig, axes


def score_distance_profiles(
    labels: Sequence[int | str],
    exact: Sequence[float],
    per_opportunity: Sequence[float],
    *,
    exact_interval: tuple[Any, Any] | None = None,
    per_opportunity_interval: tuple[Any, Any] | None = None,
    channel: str,
    grouped: bool = False,
    theme: FigureTheme = FigureTheme(),
):
    import matplotlib.pyplot as plt

    positions = np.arange(len(labels))
    with publication_style(theme):
        fig, axes = plt.subplots(
            1, 2, figsize=(theme.width * 1.75, theme.height), constrained_layout=True
        )
        for ax, values, interval, title in (
            (axes[0], exact, exact_interval, _exact_panel_title(grouped)),
            (
                axes[1],
                per_opportunity,
                per_opportunity_interval,
                "Contribution / event-carrier support",
            ),
        ):
            values = np.asarray(values)
            ax.plot(
                positions,
                values,
                marker="o",
                color=(
                    theme.semantic_color if channel == "semantic" else theme.structural_color
                ),
                linewidth=theme.line_width,
            )
            if interval is not None:
                ax.fill_between(
                    positions,
                    np.asarray(interval[0]),
                    np.asarray(interval[1]),
                    color=(
                        theme.semantic_color
                        if channel == "semantic"
                        else theme.structural_color
                    ),
                    alpha=0.20,
                    linewidth=0,
                )
            ax.set_title(title)
            ax.set_xlabel("Carrier distance from changed node")
            distance_ticks(ax, labels, theme)
            ax.grid(alpha=theme.grid_alpha, linewidth=0.5)
        fig.suptitle(f"{channel.capitalize()} donor-swap score distance profile")
    return fig, axes


def distance_support_profile(
    labels: Sequence[int | str],
    graphs: Sequence[int],
    pairs: Sequence[int],
    *,
    empty_replicate_fraction: Sequence[float] | None = None,
    minimum_graphs: int,
    minimum_pairs: int,
    channel: str,
    theme: FigureTheme = FigureTheme(),
):
    """Show how much evidence each distance column carries, against the reporting floor."""

    import matplotlib.pyplot as plt

    positions = np.arange(len(labels))
    graphs = np.asarray(graphs, dtype=np.float64)
    pairs = np.asarray(pairs, dtype=np.float64)
    colour = theme.semantic_color if channel == "semantic" else theme.structural_color
    with publication_style(theme):
        fig, axes = plt.subplots(
            1, 2, figsize=(theme.width * 1.75, theme.height), constrained_layout=True
        )
        for ax, values, floor, title in (
            (axes[0], graphs, int(minimum_graphs), "Supporting graphs"),
            (axes[1], pairs, int(minimum_pairs), "Eligible carrier-source pairs"),
        ):
            below = values < float(floor)
            ax.bar(
                positions,
                values,
                color=[theme.inactive_color if flag else colour for flag in below],
                width=0.8,
            )
            ax.axhline(
                float(floor),
                color="black",
                linewidth=theme.line_width,
                linestyle="--",
            )
            ax.annotate(
                f"reporting floor = {int(floor)}",
                xy=(0.02, float(floor)),
                xycoords=("axes fraction", "data"),
                va="bottom",
                fontsize=theme.tick_size,
            )
            ax.set_title(title)
            ax.set_xlabel("Carrier distance from changed node")
            distance_ticks(ax, labels, theme)
            ax.grid(alpha=theme.grid_alpha, linewidth=0.5, axis="y")
            if values.size and float(np.max(values)) > 0:
                ax.set_ylim(0, float(np.max(values)) * 1.18)
        if empty_replicate_fraction is not None:
            fraction = np.asarray(empty_replicate_fraction, dtype=np.float64)
            twin = axes[0].twinx()
            twin.plot(
                positions,
                fraction,
                color="black",
                linewidth=theme.line_width,
                marker="o",
                markersize=3.0,
                linestyle=":",
            )
            twin.set_ylim(0.0, 1.0)
            twin.set_ylabel("Bootstrap replicates with no support")
        fig.suptitle(
            f"{channel.capitalize()} distance-column support "
            "(grey columns are suppressed in reported figures)"
        )
    return fig, axes


def attention_distance_profiles(
    labels: Sequence[int | str],
    profiles: Mapping[str, Sequence[float]],
    *,
    grouped: bool = False,
    theme: FigureTheme = FigureTheme(),
):
    import matplotlib.pyplot as plt

    positions = np.arange(len(labels))
    with publication_style(theme):
        fig, ax = plt.subplots(figsize=(theme.width * 1.25, theme.height))
        for family, values in profiles.items():
            ax.plot(
                positions,
                values,
                linewidth=theme.line_width,
                markersize=4.0,
                markeredgecolor="white",
                markeredgewidth=0.5,
                label=target_label(family),
                **family_style(family, theme),
            )
        ax.set_xlabel("Sender–receiver shortest path-distance")
        ax.set_ylabel(
            "Clean attention mass per unit distance"
            if grouped
            else "Fraction of clean attention mass"
        )
        distance_ticks(ax, labels, theme)
        ax.set_title("Clean attention distance profile")
        ax.grid(alpha=theme.grid_alpha, linewidth=0.5)
        ax.legend(frameon=False, fontsize=theme.tick_size)
    return fig, ax


def carriage_profiles(
    x: Sequence[Any],
    functional: Sequence[float],
    beneficial: Sequence[float] | None,
    *,
    functional_interval: tuple[Any, Any] | None = None,
    beneficial_interval: tuple[Any, Any] | None = None,
    channel: str,
    theme: FigureTheme = FigureTheme(),
):
    import matplotlib.pyplot as plt

    positions = np.arange(len(x))
    with publication_style(theme):
        columns = 1 if beneficial is None else 2
        fig, axes = plt.subplots(
            1,
            columns,
            figsize=(theme.width * (1.0 if columns == 1 else 1.75), theme.height),
            constrained_layout=True,
            squeeze=False,
        )
        panels = [
            (
                axes[0, 0],
                functional,
                functional_interval,
                "Functional carriage",
                theme.functional_color,
            ),
        ]
        if beneficial is not None:
            panels.append(
                (
                    axes[0, 1],
                    beneficial,
                    beneficial_interval,
                    "Beneficial carriage",
                    theme.beneficial_color,
                )
            )
        for ax, values, interval, name, color in panels:
            values = np.asarray(values)
            ax.plot(positions, values, marker="o", color=color, linewidth=theme.line_width)
            if interval is not None:
                ax.fill_between(
                    positions,
                    np.asarray(interval[0]),
                    np.asarray(interval[1]),
                    color=color,
                    alpha=0.20,
                    linewidth=0,
                )
            if name == "Beneficial carriage":
                ax.axhline(0, color="#777777", linewidth=0.8)
            ax.set_title(name)
            ax.set_xlabel("Carrier distance from changed node")
            ax.set_xticks(positions, [str(value) for value in x])
            ax.grid(alpha=theme.grid_alpha, linewidth=0.5)
        fig.suptitle(f"{channel.capitalize()} donor-swap")
    return fig, axes[0]


def statistic_caption(statistic: Mapping[str, Any] | None) -> str:
    """One-line rank-correlation caption: rho with its interval, permutation p, and n.

    The interval is the registered nested bootstrap and the p is the within-layer permutation test,
    so the caption reports the layer-confound-controlled evidence rather than a naive pooled p.
    """

    if not statistic:
        return ""
    rho = statistic.get("rho")
    if rho is None or not np.isfinite(rho):
        return "ρ not estimable"
    parts = [f"ρ = {float(rho):.2f}"]
    low, high = statistic.get("low"), statistic.get("high")
    if low is not None and high is not None and np.isfinite(low) and np.isfinite(high):
        parts[0] += f" [{float(low):.2f}, {float(high):.2f}]"
    p = statistic.get("p")
    if p is not None and np.isfinite(p):
        parts.append("p < 0.001" if float(p) < 0.001 else f"p = {float(p):.3f}")
    n = statistic.get("n")
    if n:
        parts.append(f"n = {int(n)}")
    return ", ".join(parts)


def causal_scatter_grid(
    panels: Sequence[Mapping[str, Any]],
    *,
    theme: FigureTheme = FigureTheme(),
):
    """Publication scatter grid with two-axis percentile intervals and rank correlations.

    Each panel carries its own rank correlation above the axes, and the grid carries one shared
    key: the layer colourbar that the point colours have always encoded, and a marker for heads
    below the activity floor. Neither was documented anywhere in the figure before.
    """

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    columns = 3
    rows = int(np.ceil(len(panels) / columns))
    layer_max = max(
        (int(np.asarray(panel.get("layer", [0])).max(initial=0)) for panel in panels),
        default=0,
    )
    normalizer = plt.Normalize(0, max(1, layer_max))
    any_inactive = False
    with publication_style(theme):
        fig, axes = plt.subplots(
            rows,
            columns,
            figsize=(theme.width * 2.2, theme.height * rows),
            constrained_layout=True,
            squeeze=False,
        )
        for ax, panel in zip(axes.reshape(-1), panels):
            x = np.asarray(panel["x"])
            y = np.asarray(panel["y"])
            active = np.asarray(panel.get("active", np.ones(len(x), dtype=bool)))
            layer = np.asarray(panel.get("layer", np.zeros(len(x), dtype=int)))
            colours = plt.get_cmap(theme.layer_cmap)(normalizer(layer))
            colours[~active] = __import__("matplotlib").colors.to_rgba(
                theme.inactive_color
            )
            any_inactive = any_inactive or not bool(active.all())
            x_interval = panel.get("x_interval")
            y_interval = panel.get("y_interval")
            _errorbars(ax, x, y, x_interval, y_interval, theme)
            ax.scatter(
                x,
                y,
                c=colours,
                s=theme.marker_size * 0.82,
                edgecolor="white",
                linewidth=0.4,
            )
            ax.set_xlabel(panel["xlabel"])
            ax.set_ylabel(panel["ylabel"])
            caption = statistic_caption(panel.get("statistic"))
            title = panel.get("title", "")
            ax.set_title(
                f"{title}\n{caption}" if caption else title,
                fontsize=theme.title_size * 0.92,
            )
            if panel.get("zero_x"):
                ax.axvline(0, color="#777777", linestyle="--", linewidth=0.8)
            if panel.get("zero_y"):
                ax.axhline(0, color="#777777", linestyle="--", linewidth=0.8)
            ax.grid(alpha=theme.grid_alpha, linewidth=0.5)
        used = list(axes.reshape(-1)[: len(panels)])
        for ax in axes.reshape(-1)[len(panels) :]:
            ax.set_visible(False)
        scalar = plt.cm.ScalarMappable(norm=normalizer, cmap=theme.layer_cmap)
        bar = fig.colorbar(scalar, ax=used, pad=0.015, fraction=0.03)
        bar.set_label("Layer")
        bar.set_ticks(np.arange(layer_max + 1))
        key = [
            Line2D(
                [],
                [],
                linestyle="none",
                marker="o",
                markersize=5,
                markerfacecolor=theme.inactive_color,
                markeredgecolor="white",
                label=f"below the activity floor",
            )
        ] if any_inactive else []
        key.append(
            Line2D(
                [],
                [],
                color="#555555",
                alpha=theme.error_alpha + 0.3,
                linewidth=1.1,
                label="95% nested percentile interval",
            )
        )
        fig.legend(
            handles=key,
            frameon=False,
            fontsize=theme.tick_size,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=len(key),
        )
    return fig, axes


def family_style(name: str, theme: FigureTheme) -> dict[str, Any]:
    """Colour, marker, and dash for a frozen family, consistent across every figure.

    The leaning families take their channel's colour — a semantic-leaning family drawn in the
    structural channel's blue reads as a contradiction against the rest of the figure set — and the
    reference families take neutral greys. Marker and dash carry the same distinction so the figure
    survives greyscale printing.
    """

    key = str(name).removeprefix("family_")
    styles = {
        "semantic_leaning": (theme.semantic_color, "o", "-"),
        "structural_leaning": (theme.structural_color, "s", "--"),
        "central_responsive": (theme.central_color, "^", ":"),
        "inactive": (theme.inactive_color, "v", "-."),
    }
    colour, marker, dash = styles.get(key, (theme.central_color, "o", "-"))
    return {"color": colour, "marker": marker, "linestyle": dash}


def target_label(name: str) -> str:
    """Readable axis label for a frozen family or matched-control target."""

    text = str(name)
    for prefix in ("family_", "control_"):
        text = text.removeprefix(prefix)
    text = text.removesuffix("_control")
    for kind in ("central", "inactive", "random"):
        text = text.replace(f"_{kind}", f" / {kind}")
    return text.replace("_", " ")


def causal_family_panels(
    family_names: Sequence[str],
    values: Mapping[str, np.ndarray],
    *,
    intervals: Mapping[str, tuple[np.ndarray, np.ndarray]] | None = None,
    theme: FigureTheme = FigureTheme(),
):
    """Channel-by-family restoration/injection, alignment, and necessity endpoints.

    Targets run down a shared vertical axis rather than across five rotated category axes: the
    names are long, and one legible copy of them beats five illegible ones. Panel height follows
    the target count, so the bars keep a constant thickness however many targets are passed.
    """

    import matplotlib.pyplot as plt

    positions = np.arange(len(family_names))
    height = 0.36
    panels = (
        ("restoration_gross", "Gross restoration"),
        ("injection_gross", "Gross injection"),
        ("rescue", "Causal rescue"),
        ("induction", "Causal induction"),
        ("necessity", "Donor-wise necessity"),
    )
    figure_height = max(theme.height, 0.42 * len(family_names) + 1.5)
    with publication_style(theme):
        fig, axes = plt.subplots(
            1,
            len(panels),
            figsize=(theme.width * 2.6, figure_height),
            constrained_layout=True,
            sharey=True,
        )
        for ax, (key, title) in zip(axes, panels):
            matrix = np.asarray(values[key])  # [channel,family]
            for channel, offset, color in (
                (0, height / 2, theme.semantic_color),
                (1, -height / 2, theme.structural_color),
            ):
                error = None
                if intervals and key in intervals:
                    low, high = intervals[key]
                    error = np.vstack(
                        (
                            matrix[channel] - np.asarray(low)[channel],
                            np.asarray(high)[channel] - matrix[channel],
                        )
                    )
                ax.barh(
                    positions + offset,
                    matrix[channel],
                    height,
                    xerr=error,
                    color=color,
                    alpha=0.88,
                    capsize=2,
                    label=("Semantic donor-swap" if channel == 0 else "Structural donor-swap"),
                )
            ax.axvline(0, color="#777777", linewidth=0.8)
            ax.set_title(title)
            ax.grid(axis="x", alpha=theme.grid_alpha, linewidth=0.5)
        # Top-to-bottom in the order given, which keeps a family next to its own controls.
        axes[0].set_yticks(positions, [target_label(name) for name in family_names])
        axes[0].invert_yaxis()
        handles, entries = axes[0].get_legend_handles_labels()
        # Below the panels: an in-axes key would sit on top of whichever bar happens to be short.
        fig.legend(
            handles,
            entries,
            frameon=False,
            fontsize=theme.tick_size,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=2,
        )
    return fig, axes


def cumulative_prefix_curves(
    curves: Mapping[str, Mapping[str, Any]],
    *,
    theme: FigureTheme = FigureTheme(),
):
    """Cumulative frozen-family prefix curves with channel-specific intervals.

    A `control` entry, when present, is the size-matched control ladder at the same prefix sizes.
    It is drawn as a faint reference line rather than as its own categorical figure: the question a
    prefix ladder answers is whether the family separates from its matched control as heads
    accumulate, which is only legible when the two run on one pair of axes.
    """

    import matplotlib.pyplot as plt

    with publication_style(theme):
        fig, axes = plt.subplots(
            1, 2, figsize=(theme.width * 1.8, theme.height), constrained_layout=True
        )
        for ax, endpoint, title in (
            (axes[0], "gross", r"Gross patch response $G_c$"),
            (axes[1], "necessity", "Donor-wise necessity"),
        ):
            for family, record in curves.items():
                x = np.asarray(record["prefix"])
                for channel, color, marker in (
                    ("semantic", theme.semantic_color, "o"),
                    ("structural", theme.structural_color, "s"),
                ):
                    values = np.asarray(record[endpoint][channel])
                    low, high = record[f"{endpoint}_interval"][channel]
                    label = (
                        f"{family.replace('_', ' ')} — {channel} donor-swap"
                    )
                    ax.plot(
                        x,
                        values,
                        color=color,
                        marker=marker,
                        linestyle=("-" if family.startswith("semantic") else "--"),
                        linewidth=theme.line_width,
                        label=label,
                    )
                    ax.fill_between(
                        x,
                        np.asarray(low),
                        np.asarray(high),
                        color=color,
                        alpha=0.12,
                        linewidth=0,
                    )
                    control = (record.get("control") or {}).get(endpoint, {}).get(channel)
                    if control is not None:
                        ax.plot(
                            x,
                            np.asarray(control),
                            color=color,
                            marker=marker,
                            markersize=3.0,
                            markerfacecolor="white",
                            linestyle=":",
                            linewidth=theme.line_width * 0.8,
                            alpha=0.75,
                            label=f"{label} (matched control)",
                        )
            ax.axhline(0, color="#777777", linewidth=0.8)
            ax.set_xlabel("Cumulative frozen-family prefix size")
            ax.set_title(title)
            ax.grid(alpha=theme.grid_alpha, linewidth=0.5)
        # Adding control ladders doubles the series, so the key goes below the panels rather than
        # over the curves it describes.
        handles, entries = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            entries,
            frameon=False,
            fontsize=theme.tick_size - 1,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=2,
        )
    return fig, axes


@dataclass
class FigureBuilder:
    output_dir: Path
    theme: FigureTheme = field(default_factory=FigureTheme)
    modifier: Callable[[str, Any, Any], None] | None = None
    common_metadata: Mapping[str, Any] = field(default_factory=dict)

    def save(
        self,
        name: str,
        fig: Any,
        axes: Any,
        *,
        metadata: Mapping[str, Any],
    ) -> tuple[Path, ...]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.modifier is not None:
            self.modifier(name, fig, axes)
        paths: list[Path] = []
        for suffix in self.theme.formats:
            path = self.output_dir / f"{name}.{suffix}"
            # `publication_style` sets savefig.bbox, but its rc context closed when the figure
            # function returned, so the bound has to be given here or anything drawn outside the
            # axes -- a legend below the panels, a rotated tick label -- is cropped away silently.
            fig.savefig(
                path,
                dpi=self.theme.dpi,
                bbox_inches="tight",
                pad_inches=0.04,
            )
            paths.append(path)
        atomic_json(
            self.output_dir / f"{name}.metadata.json",
            {
                "figure": name,
                "theme": dataclasses.asdict(self.theme),
                **dict(self.common_metadata),
                **dict(metadata),
            },
        )
        import matplotlib.pyplot as plt

        plt.close(fig)
        return tuple(paths)
