"""Draw donor-swap and node-transposition interventions on one molecular graph."""

from __future__ import annotations

import argparse
import math
import os
import tempfile
from pathlib import Path

_MPLCONFIGDIR = Path(tempfile.gettempdir()) / "graph_specialisation_metrics_matplotlib"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch


INK = "#20262B"
EDGE = "#A0A9AF"
GUIDE = "#DCE1E4"
NODE = "#F2F4F5"
NODE_ALT = "#E9ECEE"
GRAY = "#6F6F6F"
RED = "#C75B5B"
BLUE = "#4C78A8"
GRAY_LIGHT = "#EFEFEF"
RED_LIGHT = "#F8ECEC"
BLUE_LIGHT = "#EAF0F6"


BASE_POSITIONS = {
    0: (-0.90, 0.58),
    1: (0.00, 1.08),
    2: (0.90, 0.58),
    3: (0.90, -0.46),
    4: (0.00, -0.96),
    5: (-0.90, -0.46),
    6: (-1.75, 1.00),
    7: (-2.55, 0.55),
    8: (1.75, 1.00),
    9: (2.55, 0.55),
    10: (0.00, -1.82),
    11: (0.85, -2.25),
    12: (1.85, 1.85),
}

EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 0),
    (0, 6),
    (6, 7),
    (2, 8),
    (8, 9),
    (8, 12),
    (4, 10),
    (10, 11),
)


def transform_positions(
    centre: tuple[float, float],
    scale: float = 0.54,
) -> dict[int, tuple[float, float]]:
    cx, cy = centre
    return {
        node: (cx + scale * x, cy + scale * y)
        for node, (x, y) in BASE_POSITIONS.items()
    }


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str,
    double: bool = False,
    rad: float = 0.0,
    linewidth: float = 1.55,
    mutation: float = 12,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="<|-|>" if double else "-|>",
            mutation_scale=mutation,
            linewidth=linewidth,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=3,
            shrinkB=3,
            capstyle="round",
            joinstyle="round",
            zorder=8,
        )
    )


def draw_node(
    ax: plt.Axes,
    position: tuple[float, float],
    *,
    fill: str,
    edge: str,
    highlighted: bool = False,
    light: str = "#FFFFFF",
) -> None:
    if highlighted:
        ax.add_patch(
            Circle(
                position,
                0.158,
                facecolor=light,
                edgecolor=edge,
                linewidth=1.25,
                zorder=4,
            )
        )
    ax.add_patch(
        Circle(
            position,
            0.108 if highlighted else 0.092,
            facecolor=fill,
            edgecolor=INK if highlighted else EDGE,
            linewidth=0.78,
            zorder=5,
        )
    )


def draw_molecule(
    ax: plt.Axes,
    centre: tuple[float, float],
    *,
    highlights: dict[int, tuple[str, str]] | None = None,
) -> dict[int, tuple[float, float]]:
    positions = transform_positions(centre)
    highlights = highlights or {}

    for source, target in EDGES:
        x0, y0 = positions[source]
        x1, y1 = positions[target]
        ax.plot(
            [x0, x1],
            [y0, y1],
            color=EDGE,
            linewidth=1.12,
            solid_capstyle="round",
            zorder=1,
        )

    for node, position in positions.items():
        if node in highlights:
            fill, light = highlights[node]
            draw_node(
                ax,
                position,
                fill=fill,
                edge=fill,
                highlighted=True,
                light=light,
            )
        else:
            draw_node(
                ax,
                position,
                fill=NODE if node % 2 == 0 else NODE_ALT,
                edge=EDGE,
            )

    return positions


def offset_towards(
    start: tuple[float, float],
    end: tuple[float, float],
    distance: float,
) -> tuple[float, float]:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy)
    return start[0] + distance * dx / length, start[1] + distance * dy / length


def draw_donor_swap(ax: plt.Axes) -> None:
    ax.text(
        2.48,
        3.08,
        "Donor swap",
        ha="center",
        va="top",
        fontsize=10.0,
        weight=500,
        color=INK,
    )

    target_node = 5
    positions = draw_molecule(
        ax,
        (2.50, 1.46),
        highlights={target_node: (RED, RED_LIGHT)},
    )
    target = positions[target_node]

    donor = (0.48, target[1])
    draw_node(
        ax,
        donor,
        fill=BLUE,
        edge=BLUE,
        highlighted=True,
        light=BLUE_LIGHT,
    )
    ax.text(
        donor[0],
        donor[1] + 0.27,
        "donor",
        ha="center",
        va="bottom",
        fontsize=7.0,
        color=BLUE,
    )

    start = offset_towards(donor, target, 0.17)
    end = offset_towards(target, donor, 0.17)
    arrow(
        ax,
        start,
        end,
        color=BLUE,
        rad=0.0,
        linewidth=1.60,
        mutation=12,
    )


def draw_node_transposition(ax: plt.Axes) -> None:
    ax.text(
        7.48,
        3.08,
        "Node transposition",
        ha="center",
        va="top",
        fontsize=10.0,
        weight=500,
        color=INK,
    )

    first_node = 0
    second_node = 3
    positions = draw_molecule(
        ax,
        (7.50, 1.46),
        highlights={
            first_node: (RED, RED_LIGHT),
            second_node: (BLUE, BLUE_LIGHT),
        },
    )
    first = positions[first_node]
    second = positions[second_node]
    start = offset_towards(first, second, 0.19)
    end = offset_towards(second, first, 0.19)
    arrow(
        ax,
        start,
        end,
        color=GRAY,
        double=True,
        linewidth=1.45,
        mutation=11,
    )


def make_figure() -> plt.Figure:
    plt.rcParams.update(
        {
            "font.family": "STIX Two Text",
            "mathtext.fontset": "stix",
            "font.size": 8.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
        }
    )

    figure, axis = plt.subplots(figsize=(8.0, 2.75))
    axis.set_xlim(0.08, 9.88)
    axis.set_ylim(0.05, 3.30)
    axis.set_aspect("equal")
    axis.axis("off")

    axis.plot(
        [4.98, 4.98],
        [0.18, 3.16],
        color=GUIDE,
        linewidth=0.72,
        zorder=0,
    )

    draw_donor_swap(axis)
    draw_node_transposition(axis)
    return figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("visualisations/generated"),
        help="Directory where the PDF and PNG will be written.",
    )
    parser.add_argument(
        "--stem",
        default="graph_interventions_donor_swap_transposition",
        help="Output filename stem.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="PNG export DPI.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    figure = make_figure()
    pdf_path = args.out_dir / f"{args.stem}.pdf"
    png_path = args.out_dir / f"{args.stem}.png"
    figure.savefig(pdf_path, bbox_inches="tight", pad_inches=0.05)
    figure.savefig(png_path, bbox_inches="tight", pad_inches=0.05, dpi=args.dpi)
    plt.close(figure)
    print(pdf_path)
    print(png_path)


if __name__ == "__main__":
    main()
