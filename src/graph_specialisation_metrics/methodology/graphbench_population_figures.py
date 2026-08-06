"""Focused four-seed GraphBench causal figures from immutable CPU caches.

The training seed is the population unit throughout.  Individual heads and donor
events are first reduced within seed; they are never pooled across independently
trained models as if they were exchangeable observations.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .cache import atomic_json
from .figures import FigureBuilder, FigureTheme, publication_style
from .protocol import PROTOCOL_VERSION, MethodologyConfig

FOCUSED_GRAPHBENCH_TASK = "graphbench_bipartite_matching_hard"
SEED_MARKERS = ("o", "s", "^", "D", "P", "X", "v", "<", ">")
CHANNELS = ("semantic", "structural")
FAMILIES = ("semantic", "structural")
NECESSITY_FAMILIES = ("semantic", "structural", "j_matched_null")


def _value(record: Any, name: str) -> Any:
    if isinstance(record, Mapping):
        return record[name]
    return getattr(record, name)


def _head_name(head: Sequence[int]) -> str:
    return f"head_L{int(head[0])}_H{int(head[1])}"


def _hierarchical_mean(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    """Reduce donors, then sources, then equally weighted graphs."""

    graph_values: list[float] = []
    for graph in sorted({int(row["graph"]) for row in rows}):
        graph_rows = [row for row in rows if int(row["graph"]) == graph]
        source_values: list[float] = []
        for source in sorted({int(row["source"]) for row in graph_rows}):
            values = np.asarray(
                [
                    float(row[field])
                    for row in graph_rows
                    if int(row["source"]) == source
                ],
                dtype=np.float64,
            )
            finite = values[np.isfinite(values)]
            if finite.size:
                source_values.append(float(np.mean(finite)))
        if source_values:
            graph_values.append(float(np.mean(source_values)))
    return float(np.mean(graph_values)) if graph_values else np.nan


def _family_event_mean(
    event_records: Mapping[str, Any],
    heads: Sequence[Sequence[int]],
    channel: str,
    field: str,
) -> float:
    values = np.asarray(
        [
            _hierarchical_mean(event_records[_head_name(head)][channel], field)
            for head in heads
            if _head_name(head) in event_records
        ],
        dtype=np.float64,
    )
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else np.nan


def _hierarchical_necessity_fraction(
    rows: Sequence[Mapping[str, Any]],
    *,
    effect_floor: float,
) -> float:
    """Reduce event-wise necessity fractions over the registered hierarchy."""

    transformed = []
    for row in rows:
        event_effect = float(row["event_effect"])
        transformed.append(
            {
                **row,
                "necessity_fraction": (
                    float(row["necessity"]) / event_effect
                    if event_effect > effect_floor
                    else np.nan
                ),
            }
        )
    return _hierarchical_mean(transformed, "necessity_fraction")


def _family_necessity_mean(
    event_records: Mapping[str, Any],
    heads: Sequence[Sequence[int]],
    channel: str,
    *,
    effect_floor: float,
) -> float:
    values = np.asarray(
        [
            _hierarchical_necessity_fraction(
                event_records[_head_name(head)][channel],
                effect_floor=effect_floor,
            )
            for head in heads
            if _head_name(head) in event_records
        ],
        dtype=np.float64,
    )
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else np.nan


def _j_matched_null_heads(
    joint_sensitivity: np.ndarray,
    selected_heads: Sequence[Sequence[int]],
) -> dict[str, Any]:
    """Select non-target heads with the closest joint-sensitivity distribution."""

    selected = tuple(dict.fromkeys(tuple(map(int, head)) for head in selected_heads))
    selected_set = set(selected)
    candidates = tuple(
        (int(layer), int(head))
        for layer, head in np.ndindex(joint_sensitivity.shape)
        if (int(layer), int(head)) not in selected_set
        and np.isfinite(joint_sensitivity[layer, head])
    )
    targets = tuple(
        head for head in selected if np.isfinite(joint_sensitivity[head])
    )
    if not targets or not candidates:
        return {
            "method": "minimum-total-absolute-J assignment without replacement",
            "heads": (),
            "matches": (),
            "mean_absolute_J_gap": np.nan,
        }

    from scipy.optimize import linear_sum_assignment

    cost = np.asarray(
        [
            [
                abs(float(joint_sensitivity[target]) - float(joint_sensitivity[candidate]))
                for candidate in candidates
            ]
            for target in targets
        ],
        dtype=np.float64,
    )
    # Deterministic tie-breaking is tiny enough not to change a substantive match.
    tie_break = np.arange(len(candidates), dtype=np.float64)[None, :] * 1.0e-12
    target_positions, candidate_positions = linear_sum_assignment(cost + tie_break)
    matches = tuple(
        {
            "target": targets[target_position],
            "null": candidates[candidate_position],
            "target_J": float(joint_sensitivity[targets[target_position]]),
            "null_J": float(joint_sensitivity[candidates[candidate_position]]),
            "absolute_J_gap": float(cost[target_position, candidate_position]),
        }
        for target_position, candidate_position in zip(
            target_positions.tolist(), candidate_positions.tolist()
        )
    )
    return {
        "method": "minimum-total-absolute-J assignment without replacement",
        "heads": tuple(row["null"] for row in matches),
        "matches": matches,
        "mean_absolute_J_gap": float(
            np.mean([row["absolute_J_gap"] for row in matches])
        ),
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Small dependency-free rank implementation with average ranks for ties."""

    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def _spearman(x: Any, y: Any) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    finite = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(finite)) < 3:
        return np.nan
    x_rank = _average_ranks(x[finite])
    y_rank = _average_ranks(y[finite])
    if np.std(x_rank) == 0 or np.std(y_rank) == 0:
        return np.nan
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def _seed_population_interval(
    values: np.ndarray,
    config: MethodologyConfig,
    *,
    seed_offset: int,
) -> dict[str, Any]:
    """Mean and percentile interval from one joint resample of training seeds."""

    values = np.asarray(values, dtype=np.float64)
    complete = np.isfinite(values.reshape(len(values), -1)).all(axis=1)
    included = values[complete]
    estimate = np.mean(included, axis=0) if len(included) else np.full(values.shape[1:], np.nan)
    if len(included) < 3:
        low = high = np.full_like(estimate, np.nan, dtype=np.float64)
    else:
        rng = np.random.default_rng(int(config.bootstrap.rng_seed) + int(seed_offset))
        indices = rng.integers(
            0,
            len(included),
            size=(int(config.bootstrap.replicates), len(included)),
        )
        draws = np.mean(included[indices], axis=1)
        low = np.quantile(draws, 0.025, axis=0)
        high = np.quantile(draws, 0.975, axis=0)
    return {
        "estimate": estimate,
        "low": low,
        "high": high,
        "included_seed_mask": complete,
        "included_seed_count": int(np.sum(complete)),
        "population_unit": "training seed",
        "replicates": int(config.bootstrap.replicates),
    }


def build_graphbench_population_figure_data(
    config: MethodologyConfig,
    task_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the publication plot payloads without touching a model, dataset, or GPU."""

    ordered = sorted(task_results, key=lambda value: int(value["seed"]))
    seeds = np.asarray([int(value["seed"]) for value in ordered], dtype=np.int64)
    if not len(ordered):
        raise ValueError("population figures require at least one cached seed")

    absolute_rows = []
    mediation_rows = []
    necessity_rows = []
    head_rows: list[dict[str, Any]] = []
    pair_counts: list[int] = []
    null_matches: list[dict[str, Any]] = []
    seed_rhos: list[float] = []
    for result in ordered:
        scores = result["scores"]
        causal = result["causal"]
        focused = causal["focused_specialists"]
        pair_set = "strongest_candidates"
        if pair_set not in focused.get("pair_set_order", ()):
            absolute_rows.append(np.full((2, 2, 2), np.nan))
            mediation_rows.append(np.full(2, np.nan))
            necessity_rows.append(np.full((3, 2), np.nan))
            pair_counts.append(0)
            null_matches.append(
                {
                    "method": "minimum-total-absolute-J assignment without replacement",
                    "heads": (),
                    "matches": (),
                    "mean_absolute_J_gap": np.nan,
                }
            )
        else:
            pairs = tuple(focused["pair_sets"][pair_set]["pairs"])
            pair_counts.append(len(pairs))
            selected = {
                "semantic": tuple(pair["semantic"] for pair in pairs),
                "structural": tuple(pair["structural"] for pair in pairs),
            }
            event_records = causal["event_records"]
            absolute = np.empty((2, 2, 2), dtype=np.float64)
            for metric_position, field in enumerate(
                ("R_align_adjusted", "I_align_adjusted")
            ):
                for family_position, family in enumerate(FAMILIES):
                    for channel_position, channel in enumerate(CHANNELS):
                        absolute[metric_position, family_position, channel_position] = (
                            _family_event_mean(
                                event_records,
                                selected[family],
                                channel,
                                field,
                            )
                        )
            absolute_rows.append(absolute)

            set_position = list(focused["pair_set_order"]).index(pair_set)
            metric_order = list(focused["metric_order"])
            cell_order = list(focused["cell_order"])
            interaction_position = cell_order.index("double_difference")
            interval = focused["interval"]
            mediation_rows.append(
                np.asarray(
                    [
                        interval.estimate[
                            set_position,
                            metric_order.index(metric),
                            interaction_position,
                        ]
                        for metric in ("restoration", "injection")
                    ],
                    dtype=np.float64,
                )
            )
            joint = np.asarray(
                _value(scores["coordinates"], "joint_sensitivity"),
                dtype=np.float64,
            )
            null_match = _j_matched_null_heads(
                joint,
                selected["semantic"] + selected["structural"],
            )
            null_matches.append(null_match)
            necessity_heads = {
                **selected,
                "j_matched_null": null_match["heads"],
            }
            necessity = np.empty((3, 2), dtype=np.float64)
            for family_position, family in enumerate(NECESSITY_FAMILIES):
                for channel_position, channel in enumerate(CHANNELS):
                    necessity[family_position, channel_position] = (
                        _family_necessity_mean(
                            event_records,
                            necessity_heads[family],
                            channel,
                            effect_floor=float(config.numerical.effect_floor),
                        )
                    )
            necessity_rows.append(necessity)

        coordinates = scores["coordinates"]
        raw_semantic = np.asarray(_value(coordinates, "raw_semantic"), dtype=np.float64)
        raw_structural = np.asarray(_value(coordinates, "raw_structural"), dtype=np.float64)
        joint = np.asarray(_value(coordinates, "joint_sensitivity"), dtype=np.float64)
        selectivity = np.asarray(_value(coordinates, "selectivity"), dtype=np.float64)
        layers, heads = joint.shape
        names = [
            f"head_L{layer}_H{head}"
            for layer in range(layers)
            for head in range(heads)
        ]
        clean = np.asarray(
            [causal["clean_ablation"][name]["prediction_movement"] for name in names],
            dtype=np.float64,
        ).reshape(joint.shape)
        rho = (
            causal.get("associations", {})
            .get("J_vs_clean_prediction_movement", {})
            .get("pooled", {})
            .get("rho")
        )
        seed_rhos.append(float(rho) if rho is not None else _spearman(joint, clean))
        head_rows.append(
            {
                "seed": int(result["seed"]),
                "layer": np.broadcast_to(
                    np.arange(layers, dtype=np.int64)[:, None],
                    joint.shape,
                ).copy(),
                "raw_semantic": raw_semantic,
                "raw_structural": raw_structural,
                "joint_sensitivity": joint,
                "selectivity": selectivity,
                "clean_ablation": clean,
            }
        )

    absolute_values = np.asarray(absolute_rows, dtype=np.float64)
    mediation_values = np.asarray(mediation_rows, dtype=np.float64)
    necessity_values = np.asarray(necessity_rows, dtype=np.float64)
    rho_values = np.asarray(seed_rhos, dtype=np.float64)
    return {
        "seeds": seeds,
        "pair_counts": np.asarray(pair_counts, dtype=np.int64),
        "absolute_patching": {
            "values": absolute_values,
            "metric_order": ("restoration", "injection"),
            "family_order": FAMILIES,
            "channel_order": CHANNELS,
            "population": _seed_population_interval(
                absolute_values, config, seed_offset=101
            ),
        },
        "preferential_mediation": {
            "values": mediation_values,
            "metric_order": ("restoration", "injection"),
            "population": _seed_population_interval(
                mediation_values, config, seed_offset=103
            ),
        },
        "necessity": {
            "values": necessity_values,
            "family_order": NECESSITY_FAMILIES,
            "channel_order": CHANNELS,
            "null_matches": null_matches,
            "population": _seed_population_interval(
                necessity_values, config, seed_offset=107
            ),
        },
        "clean_ablation": {
            "seed_rho": rho_values,
            "rho_population": _seed_population_interval(
                rho_values[:, None], config, seed_offset=109
            ),
        },
        "heads": head_rows,
    }


def _theme(config: MethodologyConfig) -> FigureTheme:
    values: Mapping[str, Any] = config.figure_overrides
    if FOCUSED_GRAPHBENCH_TASK in values and isinstance(
        values[FOCUSED_GRAPHBENCH_TASK], Mapping
    ):
        values = values[FOCUSED_GRAPHBENCH_TASK]
    return FigureTheme(
        width=4.2,
        height=3.55,
        dpi=600,
        font_size=9.5,
        label_size=10.0,
        title_size=10.5,
        tick_size=8.5,
        marker_size=32.0,
        line_width=1.2,
        grid_alpha=0.14,
    ).with_overrides(values)


def _seed_handles(seeds: Sequence[int], theme: FigureTheme):
    from matplotlib.lines import Line2D

    return [
        Line2D(
            [0],
            [0],
            color=theme.central_color,
            marker=SEED_MARKERS[position % len(SEED_MARKERS)],
            linestyle="none",
            markerfacecolor="none",
            markeredgewidth=0.9,
            label=f"Seed {int(seed)}",
        )
        for position, seed in enumerate(seeds)
    ]


def _family_handles(theme: FigureTheme, *, include_null: bool = False):
    from matplotlib.lines import Line2D

    families = [
        ("Semantic-scoring heads", theme.semantic_color),
        ("Structural-scoring heads", theme.structural_color),
    ]
    if include_null:
        families.append((r"$J$-matched null heads", theme.central_color))
    return [
        Line2D(
            [0],
            [0],
            color=color,
            marker="o",
            linestyle="none",
            markeredgecolor="white",
            markeredgewidth=0.5,
            label=label,
        )
        for label, color in families
    ]


def _style_axis(ax: Any) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.14, linewidth=0.55)
    ax.set_axisbelow(True)


def _colour_intervention_ticks(ax: Any, theme: FigureTheme) -> None:
    labels = ax.get_xticklabels()
    if len(labels) >= 2:
        labels[0].set_color(theme.semantic_color)
        labels[1].set_color(theme.structural_color)


def _legend_below(
    fig: Any,
    handles: Sequence[Any],
    *,
    ncol: int,
    title: str | None = None,
    font_size: float | None = None,
    title_font_size: float | None = None,
) -> None:
    legend = fig.legend(
        handles=handles,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=ncol,
        columnspacing=1.15,
        handletextpad=0.45,
        borderaxespad=0,
        title=title,
        fontsize=font_size,
    )
    if title is not None:
        legend.get_title().set_fontsize(
            8.5 if title_font_size is None else title_font_size
        )


def _family_population_marks(
    ax: Any,
    population: Mapping[str, Any],
    theme: FigureTheme,
    *,
    include_null: bool = False,
) -> None:
    """Categorical population means and seed-bootstrap confidence intervals."""

    x = np.arange(2, dtype=np.float64)
    colors = [theme.semantic_color, theme.structural_color]
    offsets = [-0.13, 0.13]
    if include_null:
        colors = [theme.semantic_color, theme.structural_color, theme.central_color]
        offsets = [-0.18, 0.0, 0.18]
    for family_position, (color, offset) in enumerate(
        zip(colors, offsets)
    ):
        estimate = np.asarray(population["estimate"], dtype=np.float64)[
            family_position
        ]
        low = np.asarray(population["low"], dtype=np.float64)[family_position]
        high = np.asarray(population["high"], dtype=np.float64)[family_position]
        yerr = np.maximum(
            0.0,
            np.vstack((estimate - low, high - estimate)),
        )
        ax.errorbar(
            x + offset,
            estimate,
            yerr=yerr if np.isfinite(yerr).all() else None,
            fmt="o",
            color=color,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.65,
            markersize=6.4,
            linewidth=1.45,
            capsize=3.2,
            capthick=1.15,
            linestyle="none",
            zorder=5,
        )
    ax.set_xlim(-0.28, 1.28)


def _plot_absolute_patching(data: Mapping[str, Any], theme: FigureTheme):
    import matplotlib.pyplot as plt

    population = data["absolute_patching"]["population"]
    mark_theme = theme.with_overrides(
        {
            "semantic_color": "#B63B47",
            "structural_color": "#3568A8",
            "central_color": "#595959",
        }
    )
    with publication_style(theme):
        fig, axes = plt.subplots(
            1,
            3,
            figsize=(theme.width * 3.0, theme.height),
        )
        for metric_position, (ax, title) in enumerate(
            zip(
                axes[:2],
                (
                    "Restoration",
                    "Injection",
                ),
            )
        ):
            metric_population = {
                key: np.asarray(population[key])[metric_position]
                for key in ("estimate", "low", "high")
            }
            _family_population_marks(
                ax,
                metric_population,
                mark_theme,
            )
            ax.axhline(0.0, color="#7F7F7F", linewidth=0.9)
            ax.set_xticks(
                np.arange(2, dtype=np.float64),
                ("Semantic donor-swap", "Structural donor-swap"),
            )
            _colour_intervention_ticks(ax, mark_theme)
            ax.set_title(title)
            _style_axis(ax)
        axes[0].set_ylabel("Aligned output effect")

        necessity = data["necessity"]
        _family_population_marks(
            axes[2],
            necessity["population"],
            mark_theme,
            include_null=True,
        )
        axes[2].axhline(0.0, color="#7F7F7F", linewidth=0.9)
        axes[2].set_xticks(
            np.arange(2, dtype=np.float64),
            ("Semantic donor-swap", "Structural donor-swap"),
        )
        _colour_intervention_ticks(axes[2], mark_theme)
        axes[2].set_ylabel("Donor-swap effect removed (fraction)")
        axes[2].set_title("Role-specific necessity")
        _style_axis(axes[2])
        _legend_below(
            fig,
            _family_handles(mark_theme, include_null=True),
            ncol=3,
            title="Mean and 95% bootstrap CI across four seeds",
            font_size=theme.font_size * 1.25,
            title_font_size=8.5 * 1.25,
        )
        fig.subplots_adjust(
            bottom=0.27,
            left=0.065,
            right=0.995,
            top=0.88,
            wspace=0.32,
        )
    return fig, axes


def _plot_preferential_mediation(data: Mapping[str, Any], theme: FigureTheme):
    import matplotlib.pyplot as plt

    record = data["preferential_mediation"]
    population = record["population"]
    x = np.arange(2, dtype=np.float64)
    with publication_style(theme):
        fig, ax = plt.subplots(figsize=(theme.width, theme.height))
        estimate = np.asarray(population["estimate"])
        low = np.asarray(population["low"])
        high = np.asarray(population["high"])
        yerr = np.maximum(0.0, np.vstack((estimate - low, high - estimate)))
        ax.errorbar(
            x,
            estimate,
            yerr=yerr if np.isfinite(yerr).all() else None,
            fmt="o",
            color="#4F4F4F",
            markerfacecolor="#4F4F4F",
            markeredgecolor="white",
            markeredgewidth=0.55,
            markersize=6.2,
            linewidth=1.3,
            capsize=3.0,
            capthick=1.0,
            linestyle="none",
            zorder=5,
        )
        ax.axhline(0.0, color="#9A9A9A", linewidth=0.75)
        ax.set_xticks(x, ("Restoration", "Injection"))
        ax.set_ylabel(r"Intervention-specific mediation, $\Delta\Delta$")
        ax.set_title("Causal mediation contrasts")
        _style_axis(ax)
        from matplotlib.lines import Line2D

        mean_handle = Line2D(
            [0],
            [0],
            color="#4F4F4F",
            marker="o",
            linestyle="none",
            label="Mean and 95% bootstrap CI across four seeds",
        )
        _legend_below(
            fig,
            [mean_handle],
            ncol=1,
        )
        fig.subplots_adjust(bottom=0.27, left=0.17, right=0.98, top=0.90)
    return fig, ax


def _plot_head_scatter(
    data: Mapping[str, Any],
    theme: FigureTheme,
    *,
    x_name: str,
    y_name: str,
    xlabel: str,
    ylabel: str,
    title: str,
    guides: str | None = None,
    statistic: str | None = None,
):
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    seeds = data["seeds"]
    with publication_style(theme):
        # Fixed axes rectangles make these three panels align exactly when placed
        # side by side, independent of tick-label or data-limit differences.
        fig = plt.figure(figsize=(theme.width, theme.height))
        ax = fig.add_axes((0.17, 0.25, 0.56, 0.66))
        colorbar_ax = fig.add_axes((0.80, 0.25, 0.035, 0.66))
        maximum_layer = max(
            int(np.nanmax(np.asarray(row["layer"]))) for row in data["heads"]
        )
        layer_count = maximum_layer + 1
        cmap = plt.get_cmap("viridis", layer_count)
        norm = mpl.colors.BoundaryNorm(
            np.arange(-0.5, layer_count + 0.5, 1.0),
            cmap.N,
        )
        all_x: list[np.ndarray] = []
        all_y: list[np.ndarray] = []
        for seed_position, row in enumerate(data["heads"]):
            x = np.asarray(row[x_name]).reshape(-1)
            y = np.asarray(row[y_name]).reshape(-1)
            layer = np.asarray(row["layer"]).reshape(-1)
            finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(layer)
            all_x.append(x[finite])
            all_y.append(y[finite])
            ax.scatter(
                x[finite],
                y[finite],
                c=layer[finite],
                cmap=cmap,
                norm=norm,
                marker=SEED_MARKERS[seed_position % len(SEED_MARKERS)],
                s=theme.marker_size * 0.54,
                alpha=0.68,
                linewidths=0.22,
                edgecolors="white",
                zorder=2,
            )
        if guides == "identity":
            finite_values = np.concatenate(all_x + all_y)
            upper = max(1e-12, float(np.max(finite_values)) * 1.04)
            ax.plot(
                (0.0, upper),
                (0.0, upper),
                color="#9A9A9A",
                linewidth=0.75,
                linestyle="--",
                zorder=1,
            )
            ax.set(xlim=(0.0, upper), ylim=(0.0, upper))
        elif guides == "selectivity":
            ax.axvline(0.0, color="#9A9A9A", linewidth=0.75, zorder=1)
            ax.set_xlim(-1.02, 1.02)
            ax.set_ylim(bottom=0.0)
        elif guides == "ablation":
            ax.set_xlim(left=0.0)
            # Keep the population-correlation annotation above the observed cloud.
            ax.set_ylim(0.0, 16.0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        _style_axis(ax)
        layer_ticks = np.unique(
            np.rint(
                np.linspace(0, maximum_layer, min(layer_count, 6))
            ).astype(int)
        )
        colorbar = fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=colorbar_ax,
            ticks=layer_ticks,
        )
        colorbar.set_label("Layer")
        colorbar.outline.set_linewidth(0.55)
        colorbar.ax.tick_params(length=2.5, width=0.55)
        handles = _seed_handles(seeds, theme)
        if statistic is not None:
            ax.text(
                0.025,
                0.975,
                statistic,
                transform=ax.transAxes,
                ha="left",
                va="top",
                color=theme.central_color,
                fontsize=theme.tick_size,
            )
        _legend_below(fig, handles, ncol=4)
    return fig, ax


def render_graphbench_population_figures(
    config: MethodologyConfig,
    task_name: str,
    task_results: Sequence[Mapping[str, Any]],
) -> dict[str, list[str]]:
    """Write the focused publication suite and a compact population manifest."""

    if task_name != FOCUSED_GRAPHBENCH_TASK:
        return {}
    data = build_graphbench_population_figure_data(config, task_results)
    theme = _theme(config)
    output_dir = config.root / task_name / "population_figures"
    for suffix in ("pdf", "png", "metadata.json"):
        obsolete = output_dir / f"03_population_donor_necessity.{suffix}"
        obsolete.unlink(missing_ok=True)
    builder = FigureBuilder(
        output_dir,
        theme,
        common_metadata={
            "protocol_version": PROTOCOL_VERSION,
            "protocol_fingerprint": config.fingerprint,
            "task": task_name,
            "population_unit": "training seed",
            "seeds": data["seeds"].tolist(),
        },
        preserve_canvas=True,
    )
    saved: dict[str, list[str]] = {}

    fig, axes = _plot_absolute_patching(data, theme)
    paths = builder.save(
        "01_population_restoration_injection",
        fig,
        axes,
        metadata={
            "estimand": (
                "mismatch-adjusted aligned restoration using original head output and "
                "injection using donor-swapped head output; donors within source, "
                "sources within graph, graphs within selected-head family, then seeds; "
                "role-specific necessity is aligned donor-swap effect removed divided "
                "by donor-swap output effect"
            ),
            "pair_set": "strongest_candidates",
            "pair_counts": data["pair_counts"],
            "patching_population": data["absolute_patching"]["population"],
            "necessity_population": data["necessity"]["population"],
            "necessity_null": {
                "matching_variable": "joint sensitivity J",
                "selection_uses_causal_outcomes": False,
                "matches": data["necessity"]["null_matches"],
            },
        },
    )
    saved["absolute_restoration_injection"] = [str(path) for path in paths]

    fig, axes = _plot_preferential_mediation(data, theme)
    paths = builder.save(
        "02_population_preferential_mediation",
        fig,
        axes,
        metadata={
            "estimand": (
                "matched-J semantic-versus-structural head family by semantic-versus-"
                "structural event interaction"
            ),
            "pair_set": "strongest_candidates",
            "pair_counts": data["pair_counts"],
            "population": data["preferential_mediation"]["population"],
        },
    )
    saved["preferential_mediation"] = [str(path) for path in paths]

    rho_population = data["clean_ablation"]["rho_population"]
    rho = float(np.asarray(rho_population["estimate"]).reshape(-1)[0])
    rho_low = float(np.asarray(rho_population["low"]).reshape(-1)[0])
    rho_high = float(np.asarray(rho_population["high"]).reshape(-1)[0])
    correlation_label = rf"Mean seed $\rho$ = {rho:.2f}"
    if np.isfinite(rho_low) and np.isfinite(rho_high):
        correlation_label += rf"  [{rho_low:.2f}, {rho_high:.2f}]"
    fig, axes = _plot_head_scatter(
        data,
        theme,
        x_name="joint_sensitivity",
        y_name="clean_ablation",
        xlabel=r"Joint sensitivity, $J$",
        ylabel="Output change after head ablation",
        title="Joint sensitivity and head-ablation impact",
        guides="ablation",
        statistic=correlation_label,
    )
    paths = builder.save(
        "04_population_joint_sensitivity_clean_ablation",
        fig,
        axes,
        metadata={
            "estimand": "within-seed Spearman correlation; seed-level population mean",
            "seed_rho": data["clean_ablation"]["seed_rho"],
            "rho_population": rho_population,
            "point_color": "transformer layer",
            "point_shape": "training seed",
        },
    )
    saved["joint_sensitivity_clean_ablation"] = [str(path) for path in paths]

    fig, axes = _plot_head_scatter(
        data,
        theme,
        x_name="raw_structural",
        y_name="raw_semantic",
        xlabel="Structural score",
        ylabel="Semantic score",
        title="Semantic and structural head scores",
        guides="identity",
    )
    paths = builder.save(
        "05_population_raw_semantic_structural_scores",
        fig,
        axes,
        metadata={
            "estimand": "individual-head raw coherent output-movement scores",
            "head_alignment": "not assumed across seeds",
            "point_color": "transformer layer",
            "point_shape": "training seed",
        },
    )
    saved["raw_semantic_structural_scores"] = [str(path) for path in paths]

    fig, axes = _plot_head_scatter(
        data,
        theme,
        x_name="selectivity",
        y_name="joint_sensitivity",
        xlabel=(
            r"Head selectivity, $D_{\mathrm{rel}}$ "
            r"(structural $\leftarrow$ 0 $\rightarrow$ semantic)"
        ),
        ylabel=r"Joint sensitivity, $J$",
        title="Joint sensitivity and relative selectivity",
        guides="selectivity",
    )
    paths = builder.save(
        "06_population_joint_sensitivity_selectivity",
        fig,
        axes,
        metadata={
            "estimand": "within-seed normalized individual-head score coordinates",
            "head_alignment": "not assumed across seeds",
            "point_color": "transformer layer",
            "point_shape": "training seed",
        },
    )
    saved["joint_sensitivity_selectivity"] = [str(path) for path in paths]

    atomic_json(
        output_dir / "population_figures.json",
        {
            "task": task_name,
            "population_unit": "training seed",
            "seeds": data["seeds"],
            "pair_counts": data["pair_counts"],
            "figures": saved,
            "note": (
                "All inference first reduces heads/events within a trained seed; "
                "population intervals resample the training seeds."
            ),
        },
    )
    return saved
