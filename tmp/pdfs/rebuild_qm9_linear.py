from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import fitz
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MultipleLocator


SOURCE = Path("/Users/joshgreen/Downloads/qm9_gap_headline_fig08_functional_carriage.pdf")
OUTPUT = Path(
    "/Users/joshgreen/Documents/Graph Specialisation and Metrics/output/pdf/"
    "qm9_absolute_per_carrier_sensitivity_linear.pdf"
)
PREVIEW = Path(
    "/Users/joshgreen/Documents/Graph Specialisation and Metrics/tmp/pdfs/"
    "qm9_absolute_per_carrier_sensitivity_linear.png"
)

# Drawing indices in the supplied Matplotlib vector PDF. Each pair identifies
# the confidence polygon and mean line for one panel.
SERIES = (
    (0, 0, "Dense GRIT", "Semantic donor", 2, 85, "#C75B5B"),
    (0, 1, "Dense GRIT", "PE transposition", 101, 184, "#4C78A8"),
    (0, 2, "Dense GRIT", "Local topology switch", 198, 284, "#6F63A8"),
    (1, 0, "1-hop GRIT + VN", "Semantic donor", 296, 379, "#C75B5B"),
    (1, 1, "1-hop GRIT + VN", "PE transposition", 393, 476, "#4C78A8"),
    (1, 2, "1-hop GRIT + VN", "Local topology switch", 490, 576, "#6F63A8"),
)

# The major ticks in the source are exactly one log10 decade apart.
TOP_ONE_TENTH_Y = 43.62237548828125
BOTTOM_ONE_TENTH_Y = 287.14556884765625
DECADE_HEIGHT = 23.008575439453125


def y_to_value(y: float, row: int) -> float:
    one_tenth_y = TOP_ONE_TENTH_Y if row == 0 else BOTTOM_ONE_TENTH_Y
    exponent = -1.0 - (float(y) - one_tenth_y) / DECADE_HEIGHT
    return float(10.0**exponent)


def line_points(drawing: dict) -> tuple[np.ndarray, np.ndarray]:
    items = drawing["items"]
    points = [items[0][1], *(item[2] for item in items)]
    x = np.asarray([point.x for point in points], dtype=float)
    y = np.asarray([point.y for point in points], dtype=float)
    return x, y


def band_points(
    drawing: dict, row: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    by_x: dict[float, list[float]] = defaultdict(list)
    for item in drawing["items"]:
        for point in (item[1], item[2]):
            by_x[round(float(point.x), 5)].append(float(point.y))
    xs = np.asarray(sorted(by_x), dtype=float)
    # Smaller page y is a larger value. Repeated closing vertices are harmless.
    high = np.asarray([y_to_value(min(by_x[x]), row) for x in xs], dtype=float)
    low = np.asarray([y_to_value(max(by_x[x]), row) for x in xs], dtype=float)
    return xs, low, high


def clean_decimal(value: float, _position: int) -> str:
    if abs(value) < 5.0e-8:
        return "0"
    return f"{value:.2f}"


def main() -> None:
    document = fitz.open(SOURCE)
    drawings = document[0].get_drawings()

    extracted: list[dict[str, object]] = []
    for row, col, model, channel, band_index, line_index, color in SERIES:
        _, mean_y = line_points(drawings[line_index])
        _, low, high = band_points(drawings[band_index], row)
        mean = np.asarray([y_to_value(y, row) for y in mean_y], dtype=float)
        distances = np.arange(mean.size, dtype=int)
        if not (len(low) == len(mean) == len(high)):
            raise ValueError(
                f"source geometry mismatch for {model}/{channel}: "
                f"low={len(low)}, mean={len(mean)}, high={len(high)}"
            )
        extracted.append(
            {
                "row": row,
                "col": col,
                "model": model,
                "channel": channel,
                "color": color,
                "distance": distances,
                "mean": mean,
                "low": low,
                "high": high,
            }
        )

    max_high = max(float(np.max(item["high"])) for item in extracted)
    shared_upper = np.ceil(max_high / 0.02) * 0.02

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 12.0,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(13.1, 7.25),
        sharey=True,
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.115, top=0.82, wspace=0.22, hspace=0.42)

    for item in extracted:
        row = int(item["row"])
        col = int(item["col"])
        ax = axes[row, col]
        distance = item["distance"]
        mean = item["mean"]
        low = item["low"]
        high = item["high"]
        color = str(item["color"])

        ax.fill_between(distance, low, high, color=color, alpha=0.16, linewidth=0)
        ax.plot(
            distance,
            mean,
            color=color,
            linewidth=2.2,
            marker="o",
            markersize=5.2,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.5,
        )
        ax.set_xlim(-0.35, int(np.max(distance)) + 0.35)
        ax.set_ylim(0.0, shared_upper)
        ax.set_xticks(np.arange(0, int(np.max(distance)) + 1, 2))
        ax.yaxis.set_major_locator(MultipleLocator(0.02))
        ax.yaxis.set_major_formatter(FuncFormatter(clean_decimal))
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.8)
        ax.set_axisbelow(True)
        ax.set_xlabel("Distance to changed set (hops)")
        if row == 0:
            ax.set_title(str(item["channel"]), pad=8)

    for row, label in enumerate(("Dense GRIT", "1-hop GRIT + VN")):
        axes[row, 0].text(
            -0.12,
            1.08,
            label,
            transform=axes[row, 0].transAxes,
            va="bottom",
            ha="left",
            fontsize=11.5,
            fontweight="semibold",
            color="#333333",
        )

    fig.suptitle(
        "QM9 absolute per-carrier functional sensitivity",
        fontsize=18,
        fontweight="semibold",
        y=0.965,
    )
    fig.text(
        0.5,
        0.895,
        "Linear scale; mean with 95% graph-bootstrap confidence interval",
        ha="center",
        va="center",
        fontsize=11.5,
        color="#666666",
    )
    fig.text(
        0.026,
        0.47,
        "Absolute functional carriage per carrier",
        rotation=90,
        ha="center",
        va="center",
        fontsize=11.5,
    )
    fig.text(
        0.985,
        0.025,
        "Replotted from the supplied July 24 vector figure",
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="#777777",
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    PREVIEW.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, bbox_inches="tight")
    fig.savefig(PREVIEW, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"source={SOURCE}")
    print(f"output={OUTPUT}")
    print(f"preview={PREVIEW}")
    print(f"shared_y_max={shared_upper:.4f}")
    for item in extracted:
        means = ", ".join(f"{value:.6g}" for value in item["mean"])
        print(f"{item['model']} | {item['channel']} | {means}")


if __name__ == "__main__":
    main()
