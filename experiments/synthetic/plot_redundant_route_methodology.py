"""Render dissertation figures for the redundant local/distant route task."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.switch_backend("Agg")


BLUE = "#4477AA"
ORANGE = "#EE7733"
GREEN = "#228833"
RED = "#CC6677"
PURPLE = "#AA3377"
GREY = "#777777"
LIGHT_GREY = "#D6D6D6"
TEXT = "#222222"


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def _clean_axis(axis: Any) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(width=0.8)


def _panel_label(axis: Any, label: str) -> None:
    axis.text(
        -0.13,
        1.08,
        label,
        transform=axis.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        color=TEXT,
    )


def _box(
    axis: Any,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    edge: str = GREY,
    face: str = "white",
    linewidth: float = 1.2,
    fontsize: float = 9.0,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        linewidth=linewidth,
        edgecolor=edge,
        facecolor=face,
    )
    axis.add_patch(patch)
    axis.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=TEXT,
    )
    return patch


def _arrow(
    axis: Any,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = GREY,
    linewidth: float = 1.2,
    linestyle: str = "-",
    mutation_scale: float = 10,
) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            linewidth=linewidth,
            linestyle=linestyle,
            color=color,
            shrinkA=2,
            shrinkB=2,
        )
    )


def plot_methodology(output_dir: Path) -> tuple[Path, Path]:
    """Show the task, factorial intervention, and allowed interpretation."""

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(13.8, 4.55),
        gridspec_kw={"width_ratios": (1.25, 1.15, 1.0)},
    )
    fig.subplots_adjust(left=0.035, right=0.985, bottom=0.10, top=0.80, wspace=0.28)
    fig.suptitle(
        "Factorial Functional carriage on a redundant two-route task",
        fontsize=15.5,
        y=0.965,
        color=TEXT,
    )
    fig.text(
        0.5,
        0.895,
        "The target is locally sufficient, while dense attention can also retrieve it by a semantic-structural distant lookup",
        ha="center",
        fontsize=9.5,
        color=GREY,
    )

    # A: Task construction.
    axis = axes[0]
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    _panel_label(axis, "a")
    axis.set_title("Two correct routes", pad=8, color=TEXT)
    _box(
        axis,
        (0.03, 0.39),
        0.29,
        0.22,
        "anchor\nquery q, bank s\nlocal copy y",
        edge=GREEN,
        face="#EAF4EC",
        linewidth=1.7,
    )
    record_y = (0.78, 0.59, 0.40, 0.21)
    record_labels = (
        "key 0, bank 0\nd=1",
        "key 1, bank 0\nd=4",
        "key 0, bank 1\nd=4",
        "key 1, bank 1\nd=6, value y",
    )
    for index, (y, label) in enumerate(zip(record_y, record_labels, strict=True)):
        selected = index == 3
        _box(
            axis,
            (0.62, y - 0.075),
            0.34,
            0.15,
            label,
            edge=PURPLE if selected else LIGHT_GREY,
            face="#F8EDF4" if selected else "#F6F6F6",
            linewidth=1.7 if selected else 1.0,
            fontsize=8.2,
        )
        _arrow(
            axis,
            (0.33, 0.50),
            (0.61, y),
            color=PURPLE if selected else LIGHT_GREY,
            linewidth=2.2 if selected else 0.8,
            linestyle="-" if selected else ":",
        )
    axis.text(0.46, 0.77, "dense lookup", ha="center", color=PURPLE, fontsize=8.8)
    _box(
        axis,
        (0.05, 0.03),
        0.47,
        0.15,
        "prediction =\nlocal route + distant route",
        edge=GREY,
        face="#F6F6F6",
        fontsize=8.2,
    )
    _arrow(axis, (0.18, 0.38), (0.24, 0.19), color=GREEN, linewidth=2.0)
    _arrow(axis, (0.78, 0.135), (0.45, 0.19), color=PURPLE, linewidth=2.0)

    # B: Four-condition interaction.
    axis = axes[1]
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    _panel_label(axis, "b")
    axis.set_title("One factorial extension", pad=8, color=TEXT)
    state_positions = {
        "00": (0.10, 0.57),
        "10": (0.58, 0.57),
        "01": (0.10, 0.28),
        "11": (0.58, 0.28),
    }
    state_labels = {
        "00": "$h_{00}$\nclean",
        "10": "$h_{10}$\nsemantic donor",
        "01": "$h_{01}$\nstructural donor",
        "11": "$h_{11}$\njoint donor",
    }
    for key, position in state_positions.items():
        _box(
            axis,
            position,
            0.31,
            0.16,
            state_labels[key],
            edge=PURPLE if key == "11" else GREY,
            face="#F8EDF4" if key == "11" else "#F6F6F6",
            linewidth=1.5 if key == "11" else 1.0,
            fontsize=8.5,
        )
    _arrow(axis, (0.42, 0.65), (0.57, 0.65), color=BLUE)
    _arrow(axis, (0.42, 0.36), (0.57, 0.36), color=BLUE)
    _arrow(axis, (0.255, 0.56), (0.255, 0.45), color=ORANGE)
    _arrow(axis, (0.735, 0.56), (0.735, 0.45), color=ORANGE)
    axis.text(0.50, 0.70, "semantic", ha="center", color=BLUE, fontsize=8.2)
    axis.text(0.04, 0.505, "structural", rotation=90, va="center", color=ORANGE, fontsize=8.2)
    axis.text(
        0.5,
        0.12,
        r"$I_{v,h}=|\langle h_{00}-h_{10}-h_{01}+h_{11},\,\nabla_h f\rangle|$",
        ha="center",
        va="center",
        fontsize=10.2,
        color=TEXT,
    )
    axis.text(
        0.5,
        0.025,
        "aggregate raw interaction mass by carrier distance",
        ha="center",
        fontsize=8.5,
        color=GREY,
    )

    # C: Interpretation guardrail.
    axis = axes[2]
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    _panel_label(axis, "c")
    axis.set_title("Three separate conclusions", pad=8, color=TEXT)
    conclusions = (
        (0.69, "Local-only refit = 0", "distant route is not task-necessary", GREEN),
        (0.42, "Interaction mass increases", "conjunctive distant route is instantiated", PURPLE),
        (0.15, "Rescue and damage increase", "backup capacity and matching exposure", RED),
    )
    for y, headline, detail, colour in conclusions:
        _box(
            axis,
            (0.08, y),
            0.84,
            0.18,
            f"{headline}\n{detail}",
            edge=colour,
            face="#F7F7F7",
            linewidth=1.5,
            fontsize=8.8,
        )
    axis.text(
        0.5,
        0.045,
        "Usage and reliance are measured; necessity is tested separately.",
        ha="center",
        fontsize=8.6,
        color=TEXT,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / "redundant_route_methodology.png"
    pdf = output_dir / "redundant_route_methodology.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def _fixed_route_records(controlled: Mapping[str, Any]) -> list[dict[str, float]]:
    records: list[dict[str, float]] = []
    for condition in controlled["fixed_routes"].values():
        global_mix = float(condition["global_mix"])
        for seed in condition["seeds"]:
            records.append(
                {
                    "global_mix": global_mix,
                    "seed": float(seed["seed"]),
                    **{
                        key: float(value)
                        for key, value in seed["test"].items()
                        if isinstance(value, (int, float))
                    },
                }
            )
    return sorted(records, key=lambda row: (row["global_mix"], row["seed"]))


def _multihead_records(multihead: Mapping[str, Any]) -> list[dict[str, float]]:
    records: list[dict[str, float]] = []
    for probability, condition in multihead["conditions"].items():
        for seed in condition["seeds"]:
            measurement = seed["measurement"]
            records.append(
                {
                    "corruption": float(probability),
                    "seed": float(seed["seed"]),
                    **{
                        key: float(value)
                        for key, value in measurement.items()
                        if isinstance(value, (int, float))
                    },
                }
            )
    return sorted(records, key=lambda row: (row["corruption"], row["seed"]))


def _group_summary(
    records: Sequence[Mapping[str, float]],
    x_key: str,
    y_key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.asarray(sorted({float(row[x_key]) for row in records}), dtype=float)
    values = [
        np.asarray([float(row[y_key]) for row in records if float(row[x_key]) == x])
        for x in xs
    ]
    means = np.asarray([float(np.mean(value)) for value in values])
    standard_deviations = np.asarray(
        [float(np.std(value, ddof=1)) if value.size > 1 else 0.0 for value in values]
    )
    return xs, means, standard_deviations


def _scatter_grouped(
    axis: Any,
    records: Sequence[Mapping[str, float]],
    x_key: str,
    y_key: str,
    *,
    color: str,
    marker: str = "o",
    jitter: float = 0.0,
    alpha: float = 0.72,
) -> None:
    for row in records:
        seed_offset = (float(row["seed"]) - 1.5) * jitter
        axis.scatter(
            float(row[x_key]) + seed_offset,
            float(row[y_key]),
            s=28,
            color=color,
            marker=marker,
            edgecolor="white",
            linewidth=0.5,
            alpha=alpha,
            zorder=3,
        )


def plot_results(
    controlled: Mapping[str, Any],
    multihead: Mapping[str, Any],
    output_dir: Path,
) -> tuple[Path, Path]:
    """Plot controlled calibration and the emergent multi-head replication."""

    fixed = _fixed_route_records(controlled)
    learned = _multihead_records(multihead)
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 8.35))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.09, top=0.82, hspace=0.45, wspace=0.31)
    fig.suptitle(
        "Conditional carriage recovers redundant conjunctive route allocation",
        fontsize=15.5,
        y=0.975,
        color=TEXT,
    )
    fig.text(
        0.5,
        0.925,
        "Controlled allocation establishes the estimand; an ordinary multi-head transformer reproduces the relationship without an explicit gate",
        ha="center",
        fontsize=9.4,
        color=GREY,
    )
    fig.text(
        0.015,
        0.68,
        "Controlled\nroute allocation",
        rotation=90,
        va="center",
        ha="center",
        fontsize=9.5,
        color=GREY,
    )
    fig.text(
        0.015,
        0.27,
        "Learned multi-head\nroute allocation",
        rotation=90,
        va="center",
        ha="center",
        fontsize=9.5,
        color=GREY,
    )

    # A: known distant-route allocation -> interaction mass.
    axis = axes[0, 0]
    _panel_label(axis, "a")
    x, mean, error = _group_summary(fixed, "global_mix", "interaction_effect_mass")
    _scatter_grouped(
        axis,
        fixed,
        "global_mix",
        "interaction_effect_mass",
        color=PURPLE,
        jitter=0.004,
    )
    axis.errorbar(x, mean, yerr=error, color=PURPLE, marker="o", linewidth=2.0, capsize=3)
    clean_route = controlled["learned_clean_route"]["test_mean"]
    axis.scatter(
        clean_route["effective_global_mix"],
        clean_route["interaction_effect_mass"],
        marker="*",
        s=115,
        color=GREEN,
        edgecolor="white",
        linewidth=0.7,
        zorder=4,
        label="learned clean route",
    )
    axis.set_xlabel("Distant-route mixture")
    axis.set_ylabel("Raw interaction carriage mass")
    axis.set_title("Known route amount is recovered")
    axis.set_xlim(0, 0.82)
    axis.set_ylim(bottom=0)
    axis.legend(frameon=False, loc="upper left")
    distance_values = np.asarray(
        [row["interaction_expected_distance"] for row in fixed], dtype=float
    )
    axis.text(
        0.98,
        0.06,
        f"normalised E[d] stays {distance_values.mean():.3f}",
        transform=axis.transAxes,
        ha="right",
        color=GREY,
        fontsize=8.2,
    )
    _clean_axis(axis)

    # B: two-sided reliance interpretation.
    axis = axes[0, 1]
    _panel_label(axis, "b")
    mass = np.asarray([row["interaction_effect_mass"] for row in fixed])
    for key, label, colour, marker in (
        ("local_failure_rescue", "Local-failure rescue", GREEN, "o"),
        ("distant_failure_damage", "Distant-failure damage", RED, "^"),
    ):
        values = np.asarray([row[key] for row in fixed])
        order = np.argsort(mass)
        axis.scatter(
            mass,
            values,
            s=34,
            color=colour,
            marker=marker,
            edgecolor="white",
            linewidth=0.5,
            alpha=0.76,
            label=label,
        )
        axis.plot(mass[order], values[order], color=colour, linewidth=1.25, alpha=0.65)
    axis.set_xlabel("Raw interaction carriage mass")
    axis.set_ylabel("Change in absolute error")
    axis.set_title("One mass, two behavioural consequences")
    axis.set_xlim(left=0)
    axis.set_ylim(bottom=0)
    axis.legend(frameon=False)
    axis.text(
        0.97,
        0.06,
        "both r > 0.999",
        transform=axis.transAxes,
        ha="right",
        color=GREY,
        fontsize=8.3,
    )
    _clean_axis(axis)

    # C: frozen reliance versus retrained sufficiency.
    axis = axes[0, 2]
    _panel_label(axis, "c")
    for key, label, colour, marker in (
        ("frozen_global_removed_mae", "Remove distant route (frozen)", RED, "^"),
        ("clean_mae", "Clean model", BLUE, "o"),
        ("local_refit_mae", "Refit local-only model", GREEN, "s"),
    ):
        x, mean, error = _group_summary(fixed, "global_mix", key)
        axis.errorbar(
            x,
            mean,
            yerr=error,
            marker=marker,
            color=colour,
            linewidth=1.8,
            capsize=3,
            label=label,
        )
    axis.set_xlabel("Distant-route mixture")
    axis.set_ylabel("Mean absolute error")
    axis.set_title("Reliance is not task necessity")
    axis.set_xlim(0.18, 0.82)
    axis.set_ylim(-0.025, 0.83)
    axis.legend(frameon=False, loc="upper left")
    axis.text(
        0.98,
        0.06,
        "local-only refit remains exact",
        transform=axis.transAxes,
        ha="right",
        color=GREEN,
        fontsize=8.2,
    )
    _clean_axis(axis)

    # D: reliability manipulation -> emergent interaction mass.
    axis = axes[1, 0]
    _panel_label(axis, "d")
    x, mean, error = _group_summary(learned, "corruption", "interaction_effect_mass")
    positions = np.arange(x.size, dtype=float)
    position_lookup = {float(value): float(position) for value, position in zip(x, positions, strict=True)}
    for row in learned:
        seed_offset = (float(row["seed"]) - 1.5) * 0.045
        axis.scatter(
            position_lookup[float(row["corruption"])] + seed_offset,
            float(row["interaction_effect_mass"]),
            s=28,
            color=PURPLE,
            edgecolor="white",
            linewidth=0.5,
            alpha=0.72,
            zorder=3,
        )
    axis.errorbar(
        positions,
        mean,
        yerr=error,
        color=PURPLE,
        marker="o",
        linewidth=2.0,
        capsize=3,
    )
    axis.set_xlabel("Local-copy corruption during training")
    axis.set_ylabel("Raw interaction carriage mass")
    axis.set_title("The transformer acquires the distant route")
    axis.set_xticks(positions, [f"{100 * value:.0f}%" for value in x])
    axis.set_ylim(bottom=0)
    axis.text(
        0.04,
        0.93,
        "distant corruption fixed at 5%",
        transform=axis.transAxes,
        va="top",
        color=GREY,
        fontsize=8.2,
    )
    _clean_axis(axis)

    # E: emergent carriage retains the two-sided relationship.
    axis = axes[1, 1]
    _panel_label(axis, "e")
    mass = np.asarray([row["interaction_effect_mass"] for row in learned])
    correlation_labels = {
        "local_failure_rescue": multihead["correlations"][
            "interaction_mass_vs_local_rescue"
        ],
        "distant_failure_damage": multihead["correlations"][
            "interaction_mass_vs_distant_damage"
        ],
    }
    for key, label, colour, marker in (
        ("local_failure_rescue", "Local-failure rescue", GREEN, "o"),
        ("distant_failure_damage", "Distant-failure damage", RED, "^"),
    ):
        values = np.asarray([row[key] for row in learned])
        axis.scatter(
            mass,
            values,
            s=38,
            marker=marker,
            color=colour,
            edgecolor="white",
            linewidth=0.55,
            alpha=0.80,
            label=f"{label} (r={correlation_labels[key]:.3f})",
        )
        fit = np.polyfit(mass, values, deg=1)
        grid = np.linspace(0, float(mass.max()) * 1.04, 100)
        axis.plot(grid, np.polyval(fit, grid), color=colour, linewidth=1.4, alpha=0.75)
    axis.set_xlabel("Raw interaction carriage mass")
    axis.set_ylabel("Change in absolute error")
    axis.set_title("Aggregate carriage predicts reliance")
    axis.set_xlim(left=0)
    axis.set_ylim(bottom=0)
    axis.legend(frameon=False, loc="upper left")
    _clean_axis(axis)

    # F: semantic and structural profiles increasingly co-peak.
    axis = axes[1, 2]
    _panel_label(axis, "f")
    x_values = np.asarray(
        sorted({float(row["corruption"]) for row in learned}), dtype=float
    )
    positions = np.arange(x_values.size, dtype=float)
    position_lookup = {
        float(value): float(position)
        for value, position in zip(x_values, positions, strict=True)
    }
    for key, label, colour, marker in (
        ("head_score_alignment", "Semantic-structural alignment", PURPLE, "o"),
        ("head_relative_imbalance", "Relative score imbalance", ORANGE, "s"),
    ):
        x, mean, error = _group_summary(learned, "corruption", key)
        for row in learned:
            seed_offset = (float(row["seed"]) - 1.5) * 0.045
            axis.scatter(
                position_lookup[float(row["corruption"])] + seed_offset,
                float(row[key]),
                s=28,
                color=colour,
                marker=marker,
                edgecolor="white",
                linewidth=0.5,
                alpha=0.55,
                zorder=3,
            )
        axis.errorbar(
            positions,
            mean,
            yerr=error,
            color=colour,
            marker=marker,
            linewidth=1.8,
            capsize=3,
            label=label,
        )
    axis.set_xlabel("Local-copy corruption during training")
    axis.set_ylabel("Across-head score statistic")
    axis.set_title("Conjunctive routing produces co-peaking")
    axis.set_xticks(
        positions, [f"{100 * value:.0f}%" for value in x_values]
    )
    axis.set_ylim(0, 1.02)
    axis.legend(frameon=False, loc="center right")
    _clean_axis(axis)

    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / "redundant_route_results.png"
    pdf = output_dir / "redundant_route_results.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--controlled-results",
        type=Path,
        default=Path(
            "outputs/molecular_redundant_route_robustness_balanced/results.json"
        ),
    )
    parser.add_argument(
        "--multihead-results",
        type=Path,
        default=Path("outputs/molecular_multihead_redundancy_sum_v2/results.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/pdf/redundant_route_methodology"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, str]:
    args = build_parser().parse_args(argv)
    _configure_style()
    controlled = _load(args.controlled_results)
    multihead = _load(args.multihead_results)
    method_png, method_pdf = plot_methodology(args.output_dir)
    result_png, result_pdf = plot_results(controlled, multihead, args.output_dir)
    payload = {
        "methodology_png": str(method_png),
        "methodology_pdf": str(method_pdf),
        "results_png": str(result_png),
        "results_pdf": str(result_pdf),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


if __name__ == "__main__":  # pragma: no cover
    main()
