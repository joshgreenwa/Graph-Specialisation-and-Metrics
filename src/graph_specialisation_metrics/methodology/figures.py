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
    semantic_color: str = "#C44E52"
    structural_color: str = "#4C72B0"
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
    theme: FigureTheme = FigureTheme(),
):
    import matplotlib.pyplot as plt

    exact = np.asarray(exact)
    per_opportunity = np.asarray(per_opportunity)
    if exact.shape != per_opportunity.shape:
        raise ValueError("exact/support-normalized heatmaps must have the same geometry")
    with publication_style(theme):
        fig, axes = plt.subplots(
            1, 2, figsize=(theme.width * 1.75, theme.height), constrained_layout=True
        )
        for ax, matrix, panel in zip(
            axes,
            (exact, per_opportunity),
            ("Exact score contribution", "Contribution / event-carrier support"),
        ):
            image = ax.imshow(matrix, origin="lower", aspect="auto", cmap="magma")
            ax.set_xlabel("Carrier distance from changed node")
            ax.set_ylabel("Layer")
            ax.set_xticks(np.arange(len(labels)), [str(value) for value in labels])
            ax.set_yticks(np.arange(matrix.shape[0]))
            ax.set_title(panel)
            fig.colorbar(image, ax=ax, pad=0.02)
        fig.suptitle(title or f"{channel.capitalize()} donor-swap")
    return fig, axes


def score_distance_profiles(
    labels: Sequence[int | str],
    exact: Sequence[float],
    per_opportunity: Sequence[float],
    *,
    exact_interval: tuple[Any, Any] | None = None,
    per_opportunity_interval: tuple[Any, Any] | None = None,
    channel: str,
    theme: FigureTheme = FigureTheme(),
):
    import matplotlib.pyplot as plt

    positions = np.arange(len(labels))
    with publication_style(theme):
        fig, axes = plt.subplots(
            1, 2, figsize=(theme.width * 1.75, theme.height), constrained_layout=True
        )
        for ax, values, interval, title in (
            (axes[0], exact, exact_interval, "Exact score contribution"),
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
            ax.set_xticks(positions, [str(value) for value in labels])
            ax.grid(alpha=theme.grid_alpha, linewidth=0.5)
        fig.suptitle(f"{channel.capitalize()} donor-swap score distance profile")
    return fig, axes


def attention_distance_profiles(
    labels: Sequence[int | str],
    profiles: Mapping[str, Sequence[float]],
    *,
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
                marker="o",
                linewidth=theme.line_width,
                label=family.replace("_", " "),
            )
        ax.set_xlabel("Pristine sender–receiver distance")
        ax.set_ylabel("Fraction of clean attention mass")
        ax.set_xticks(positions, [str(value) for value in labels])
        ax.set_title("Clean attention distance profile")
        ax.grid(alpha=theme.grid_alpha, linewidth=0.5)
        ax.legend(frameon=False, fontsize=theme.tick_size)
    return fig, ax


def carriage_profiles(
    x: Sequence[Any],
    functional: Sequence[float],
    beneficial: Sequence[float],
    *,
    functional_interval: tuple[Any, Any] | None = None,
    beneficial_interval: tuple[Any, Any] | None = None,
    channel: str,
    theme: FigureTheme = FigureTheme(),
):
    import matplotlib.pyplot as plt

    positions = np.arange(len(x))
    with publication_style(theme):
        fig, axes = plt.subplots(
            1, 2, figsize=(theme.width * 1.75, theme.height), constrained_layout=True
        )
        for ax, values, interval, name, color in (
            (
                axes[0],
                functional,
                functional_interval,
                "Functional carriage",
                theme.functional_color,
            ),
            (
                axes[1],
                beneficial,
                beneficial_interval,
                "Beneficial carriage",
                theme.beneficial_color,
            ),
        ):
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
    return fig, axes


def causal_scatter_grid(
    panels: Sequence[Mapping[str, Any]],
    *,
    theme: FigureTheme = FigureTheme(),
):
    """Publication scatter grid with optional two-axis percentile intervals."""

    import matplotlib.pyplot as plt

    columns = 3
    rows = int(np.ceil(len(panels) / columns))
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
            normalizer = plt.Normalize(0, max(1, int(layer.max(initial=0))))
            colours = plt.get_cmap(theme.layer_cmap)(normalizer(layer))
            colours[~active] = __import__("matplotlib").colors.to_rgba(
                theme.inactive_color
            )
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
            ax.set_title(panel.get("title", ""))
            if panel.get("zero_x"):
                ax.axvline(0, color="#777777", linestyle="--", linewidth=0.8)
            if panel.get("zero_y"):
                ax.axhline(0, color="#777777", linestyle="--", linewidth=0.8)
            ax.grid(alpha=theme.grid_alpha, linewidth=0.5)
        for ax in axes.reshape(-1)[len(panels) :]:
            ax.set_visible(False)
    return fig, axes


def causal_family_panels(
    family_names: Sequence[str],
    values: Mapping[str, np.ndarray],
    *,
    intervals: Mapping[str, tuple[np.ndarray, np.ndarray]] | None = None,
    theme: FigureTheme = FigureTheme(),
):
    """Channel-by-family restoration/injection, alignment, and necessity endpoints."""

    import matplotlib.pyplot as plt

    positions = np.arange(len(family_names))
    width = 0.34
    with publication_style(theme):
        fig, axes = plt.subplots(
            1, 5, figsize=(theme.width * 3.5, theme.height), constrained_layout=True
        )
        for ax, key, title in zip(
            axes,
            ("restoration_gross", "injection_gross", "rescue", "induction", "necessity"),
            (
                "Gross restoration",
                "Gross injection",
                "Causal rescue",
                "Causal induction",
                "Donor-wise necessity",
            ),
        ):
            matrix = np.asarray(values[key])  # [channel,family]
            for channel, offset, color in (
                (0, -width / 2, theme.semantic_color),
                (1, width / 2, theme.structural_color),
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
                ax.bar(
                    positions + offset,
                    matrix[channel],
                    width,
                    yerr=error,
                    color=color,
                    alpha=0.88,
                    capsize=2,
                    label=("Semantic donor-swap" if channel == 0 else "Structural donor-swap"),
                )
            ax.axhline(0, color="#777777", linewidth=0.8)
            ax.set_title(title)
            ax.set_xticks(
                positions,
                [name.replace("family_", "").replace("_", " ") for name in family_names],
                rotation=25,
                ha="right",
            )
            ax.grid(axis="y", alpha=theme.grid_alpha, linewidth=0.5)
        axes[0].legend(frameon=False, fontsize=theme.tick_size)
    return fig, axes


def cumulative_prefix_curves(
    curves: Mapping[str, Mapping[str, Any]],
    *,
    theme: FigureTheme = FigureTheme(),
):
    """Cumulative frozen-family prefix curves with channel-specific intervals."""

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
            ax.axhline(0, color="#777777", linewidth=0.8)
            ax.set_xlabel("Cumulative frozen-family prefix size")
            ax.set_title(title)
            ax.grid(alpha=theme.grid_alpha, linewidth=0.5)
        axes[0].legend(frameon=False, fontsize=theme.tick_size - 1)
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
            fig.savefig(path, dpi=self.theme.dpi)
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
