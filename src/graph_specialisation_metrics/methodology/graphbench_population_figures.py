"""Focused four-seed GraphBench causal figures from immutable CPU caches.

The training seed is the population unit throughout.  Individual heads and donor
events are first reduced within seed; they are never pooled across independently
trained models as if they were exchangeable observations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .cache import atomic_json
from .figures import FigureBuilder, FigureTheme, publication_style
from .protocol import PROTOCOL_VERSION, MethodologyConfig


FOCUSED_GRAPHBENCH_TASK = "graphbench_bipartite_matching_hard"
SEED_MARKERS = ("o", "s", "^", "D", "P", "X", "v", "<", ">")
CHANNELS = ("semantic", "structural")
FAMILIES = ("semantic", "structural")


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
    """Build all six plot payloads without touching a model, dataset, or GPU."""

    ordered = sorted(task_results, key=lambda value: int(value["seed"]))
    seeds = np.asarray([int(value["seed"]) for value in ordered], dtype=np.int64)
    if not len(ordered):
        raise ValueError("population figures require at least one cached seed")

    absolute_rows = []
    mediation_rows = []
    necessity_rows = []
    head_rows: list[dict[str, Any]] = []
    pair_counts: list[int] = []
    seed_rhos: list[float] = []
    for result in ordered:
        scores = result["scores"]
        causal = result["causal"]
        focused = causal["focused_specialists"]
        pair_set = "strongest_candidates"
        if pair_set not in focused.get("pair_set_order", ()):
            absolute_rows.append(np.full((2, 2, 2), np.nan))
            mediation_rows.append(np.full(2, np.nan))
            necessity_rows.append(np.full((2, 2), np.nan))
            pair_counts.append(0)
        else:
            pairs = tuple(focused["pair_sets"][pair_set]["pairs"])
            pair_counts.append(len(pairs))
            selected = {
                "semantic": tuple(pair["semantic"] for pair in pairs),
                "structural": tuple(pair["structural"] for pair in pairs),
            }
            event_records = causal["event_records"]
            absolute = np.empty((2, 2, 2), dtype=np.float64)
            for metric_position, field in enumerate(("R_align", "I_align")):
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
            necessity_position = metric_order.index("necessity_fraction")
            necessity_rows.append(
                np.asarray(
                    interval.estimate[set_position, necessity_position, :4],
                    dtype=np.float64,
                ).reshape(2, 2)
            )

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
            "family_order": FAMILIES,
            "channel_order": CHANNELS,
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


def _family_handles(theme: FigureTheme):
    from matplotlib.lines import Line2D

    return [
        Line2D(
            [0],
            [0],
            color=color,
            marker="o",
            linestyle="none",
            markeredgecolor="white",
            markeredgewidth=0.5,
            label=f"{family.title()}-scoring heads",
        )
        for family, color in zip(
            FAMILIES,
            (theme.semantic_color, theme.structural_color),
        )
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


def _legend_below(fig: Any, handles: Sequence[Any], *, ncol: int) -> None:
    fig.legend(
        handles=handles,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=ncol,
        columnspacing=1.15,
        handletextpad=0.45,
        borderaxespad=0,
    )


def _family_population_marks(
    ax: Any,
    values: np.ndarray,
    population: Mapping[str, Any],
    seeds: Sequence[int],
    theme: FigureTheme,
) -> None:
    """Categorical four-seed points plus the seed-bootstrap population interval."""

    x = np.arange(2, dtype=np.float64)
    seed_jitter = np.linspace(-0.027, 0.027, len(seeds))
    for family_position, (color, offset) in enumerate(
        zip(
            (theme.semantic_color, theme.structural_color),
            (-0.13, 0.13),
        )
    ):
        for seed_position, _seed in enumerate(seeds):
            ax.scatter(
                x + offset + seed_jitter[seed_position],
                values[seed_position, family_position],
                marker=SEED_MARKERS[seed_position % len(SEED_MARKERS)],
                s=theme.marker_size * 0.56,
                facecolors="white",
                edgecolors=color,
                linewidths=0.85,
                alpha=0.90,
                zorder=4,
            )
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
            markeredgewidth=0.55,
            markersize=6.0,
            linewidth=1.25,
            capsize=3.0,
            capthick=1.0,
            linestyle="none",
            zorder=5,
        )
    ax.set_xlim(-0.28, 1.28)


def _plot_family_by_channel(
    values: np.ndarray,
    population: Mapping[str, Any],
    seeds: Sequence[int],
    theme: FigureTheme,
    *,
    ylabel: str,
    title: str,
):
    import matplotlib.pyplot as plt

    with publication_style(theme):
        fig, ax = plt.subplots(figsize=(theme.width, theme.height))
        _family_population_marks(ax, values, population, seeds, theme)
        ax.axhline(0.0, color="#9A9A9A", linewidth=0.75)
        ax.set_xticks(
            np.arange(2, dtype=np.float64),
            ("Semantic intervention", "Structural intervention"),
        )
        _colour_intervention_ticks(ax, theme)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        _style_axis(ax)
        _legend_below(
            fig,
            _family_handles(theme) + _seed_handles(seeds, theme),
            ncol=3,
        )
        fig.subplots_adjust(bottom=0.27, left=0.16, right=0.98, top=0.90)
    return fig, ax


def _plot_absolute_patching(data: Mapping[str, Any], theme: FigureTheme):
    import matplotlib.pyplot as plt

    values = np.asarray(data["absolute_patching"]["values"])
    population = data["absolute_patching"]["population"]
    seeds = data["seeds"]
    with publication_style(theme):
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(theme.width * 2.0, theme.height),
            sharey=True,
        )
        for metric_position, (ax, title) in enumerate(
            zip(
                axes,
                ("Causal restoration", "Causal injection"),
            )
        ):
            metric_population = {
                key: np.asarray(population[key])[metric_position]
                for key in ("estimate", "low", "high")
            }
            _family_population_marks(
                ax,
                values[:, metric_position],
                metric_population,
                seeds,
                theme,
            )
            ax.axhline(0.0, color="#9A9A9A", linewidth=0.75)
            ax.set_xticks(
                np.arange(2, dtype=np.float64),
                ("Semantic intervention", "Structural intervention"),
            )
            _colour_intervention_ticks(ax, theme)
            ax.set_title(title)
            _style_axis(ax)
        axes[0].set_ylabel("Aligned change in model output")
        _legend_below(
            fig,
            _family_handles(theme) + _seed_handles(seeds, theme),
            ncol=6,
        )
        fig.subplots_adjust(bottom=0.23, left=0.09, right=0.99, top=0.89, wspace=0.16)
    return fig, axes


def _plot_preferential_mediation(data: Mapping[str, Any], theme: FigureTheme):
    import matplotlib.pyplot as plt

    record = data["preferential_mediation"]
    values = np.asarray(record["values"])
    population = record["population"]
    seeds = data["seeds"]
    x = np.arange(2, dtype=np.float64)
    with publication_style(theme):
        fig, ax = plt.subplots(figsize=(theme.width, theme.height))
        seed_jitter = np.linspace(-0.035, 0.035, len(seeds))
        for seed_position, _seed in enumerate(seeds):
            ax.scatter(
                x + seed_jitter[seed_position],
                values[seed_position],
                marker=SEED_MARKERS[seed_position % len(SEED_MARKERS)],
                s=theme.marker_size * 0.62,
                facecolors="white",
                edgecolors=theme.central_color,
                linewidths=0.85,
                alpha=0.9,
                zorder=4,
            )
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
            label="Four-seed mean (95% CI)",
        )
        _legend_below(
            fig,
            _seed_handles(seeds, theme) + [mean_handle],
            ncol=3,
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
    import matplotlib.pyplot as plt

    seeds = data["seeds"]
    with publication_style(theme):
        fig, ax = plt.subplots(figsize=(theme.width, theme.height))
        all_x: list[np.ndarray] = []
        all_y: list[np.ndarray] = []
        for seed_position, row in enumerate(data["heads"]):
            x = np.asarray(row[x_name]).reshape(-1)
            y = np.asarray(row[y_name]).reshape(-1)
            finite = np.isfinite(x) & np.isfinite(y)
            all_x.append(x[finite])
            all_y.append(y[finite])
            ax.scatter(
                x[finite],
                y[finite],
                marker=SEED_MARKERS[seed_position % len(SEED_MARKERS)],
                s=theme.marker_size * 0.46,
                color="#666666",
                alpha=0.48,
                linewidths=0.25,
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
            ax.set_aspect("equal", adjustable="box")
        elif guides == "selectivity":
            ax.axvline(0.0, color="#9A9A9A", linewidth=0.75, zorder=1)
            ax.set_xlim(-1.02, 1.02)
            ax.set_ylim(bottom=0.0)
        elif guides == "ablation":
            ax.set_xlim(left=0.0)
            ax.set_ylim(bottom=0.0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        _style_axis(ax)
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
        bottom = {
            "identity": 0.29,
            "selectivity": 0.32,
        }.get(guides, 0.22)
        fig.subplots_adjust(
            bottom=bottom,
            left=0.16,
            right=0.98,
            top=0.90,
        )
    return fig, ax


def render_graphbench_population_figures(
    config: MethodologyConfig,
    task_name: str,
    task_results: Sequence[Mapping[str, Any]],
) -> dict[str, list[str]]:
    """Write the focused six-figure suite and a compact population manifest."""

    if task_name != FOCUSED_GRAPHBENCH_TASK:
        return {}
    data = build_graphbench_population_figure_data(config, task_results)
    theme = _theme(config)
    output_dir = config.root / task_name / "population_figures"
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
                "raw matched aligned patch response; donors within source, sources "
                "within graph, graphs within selected-head family, then seeds"
            ),
            "pair_set": "strongest_candidates",
            "pair_counts": data["pair_counts"],
            "population": data["absolute_patching"]["population"],
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

    fig, axes = _plot_family_by_channel(
        data["necessity"]["values"],
        data["necessity"]["population"],
        data["seeds"],
        theme,
        ylabel="Fraction of intervention effect removed",
        title="Head necessity by intervention type",
    )
    paths = builder.save(
        "03_population_donor_necessity",
        fig,
        axes,
        metadata={
            "estimand": "aligned necessity divided by the donor event output effect",
            "pair_set": "strongest_candidates",
            "pair_counts": data["pair_counts"],
            "population": data["necessity"]["population"],
        },
    )
    saved["donor_necessity"] = [str(path) for path in paths]

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
        },
    )
    saved["raw_semantic_structural_scores"] = [str(path) for path in paths]

    fig, axes = _plot_head_scatter(
        data,
        theme,
        x_name="selectivity",
        y_name="joint_sensitivity",
        xlabel=(
            "Relative selectivity, "
            r"$D_{\mathrm{rel}}$"
            "\n"
            r"$\leftarrow$ structural  |  semantic $\rightarrow$"
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
