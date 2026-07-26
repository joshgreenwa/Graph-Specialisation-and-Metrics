"""Draw donor-swap and node-transposition interventions on one molecular graph."""

from __future__ import annotations

import argparse
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

_MPLCONFIGDIR = Path(tempfile.gettempdir()) / "graph_specialisation_metrics_matplotlib"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Polygon


# Nature sets its figures in Helvetica.  Arial is the metric-compatible stand
# in that actually ships bold and italic faces on macOS -- Helvetica is
# registered as a single regular face, so bold panel letters and italic maths
# silently fall back to roman.  Setting the mathtext fonts explicitly keeps the
# equations in the same family as the labels instead of a serif STIX fallback.
FONT_STACK = ("Arial", "Helvetica Neue", "Helvetica", "DejaVu Sans")
MATH_FONT = "Arial"

INK = "#16191C"
MUTED = "#5B646B"
BOND = "#9AA4AB"
RING = "#EEF2F4"
NODE_FILL = "#FFFFFF"
NODE_EDGE = "#9AA4AB"
ARROW = "#454E54"

# One accent hue: purple marks every node the intervention touches inside the
# graph, so the site v reads the same in both panels.  The donor sits outside
# the graph and takes a near-white tint of the same hue (both sit at ~257 deg),
# which needs a coloured stroke and symbol to register at all.  A single hue is
# inherently colour-vision safe, so what matters is contrast against the neutral
# structure: dE 50 vs the bonds, 40 vs the arrow grey, and 7.4:1 against white
# where the hue is used for text.
PURPLE = "#5C4A8E"
PURPLE_PALE = "#E4DEF1"

# Ideal bond geometry: unit bonds, 120 degree angles, two fused six-rings plus
# three substituents.  Left of the ring system is kept clear so both panels can
# stage their intervention in the same region.
ROOT3 = math.sqrt(3.0)

ATOMS: dict[str, tuple[float, float]] = {
    "a1": (0.5 * ROOT3, 0.5),
    "a2": (0.0, 1.0),
    "a3": (-0.5 * ROOT3, 0.5),
    "a4": (-0.5 * ROOT3, -0.5),
    "a5": (0.0, -1.0),
    "a6": (0.5 * ROOT3, -0.5),
    "b1": (1.5 * ROOT3, 0.5),
    "b2": (ROOT3, 1.0),
    "b5": (ROOT3, -1.0),
    "b6": (1.5 * ROOT3, -0.5),
    "s1": (0.0, 2.0),
    "s2": (2.0 * ROOT3, 1.0),
    "s3": (ROOT3, -2.0),
}

BONDS = (
    ("a1", "a2"),
    ("a2", "a3"),
    ("a3", "a4"),
    ("a4", "a5"),
    ("a5", "a6"),
    ("a6", "a1"),
    ("a1", "b2"),
    ("b2", "b1"),
    ("b1", "b6"),
    ("b6", "b5"),
    ("b5", "a6"),
    ("a2", "s1"),
    ("b1", "s2"),
    ("b5", "s3"),
)

RINGS = (
    ("a1", "a2", "a3", "a4", "a5", "a6"),
    ("a1", "b2", "b1", "b6", "b5", "a6"),
)

TARGET = "a4"
PARTNER = "s3"
DONOR = (-2.62, -0.5)
SWAP_RAD = 0.50

R_PLAIN = 0.190
R_MARK = 0.330
R_GAP = 1.10

FIG_WIDTH = 8.0
X_SPAN = 10.0
PT_PER_UNIT = 72.0 * FIG_WIDTH / X_SPAN

MARGIN = 0.16
GUTTER = 1.10


def content_box(*, donor: bool) -> tuple[tuple[float, float], tuple[float, float]]:
    """Extent of everything a panel draws, in molecule units.

    Marked nodes are measured at their full drawn radius, so a fitted molecule
    can never push a highlight into the header or the caption.  Only the donor
    panel pays for the detached node, which keeps the second panel from
    carrying an empty column where that node would have been.
    """

    xs: list[float] = []
    ys: list[float] = []
    marked = {TARGET, PARTNER}
    for name, (x, y) in ATOMS.items():
        radius = R_MARK if name in marked else R_PLAIN
        xs.extend((x - radius, x + radius))
        ys.extend((y - radius, y + radius))
    if donor:
        xs.extend((DONOR[0] - R_MARK, DONOR[0] + R_MARK))
        ys.extend((DONOR[1] - R_MARK, DONOR[1] + R_MARK))
    pad = 0.08
    return (min(xs) - pad, max(xs) + pad), (min(ys) - pad, max(ys) + pad)


BOX_DONOR_X, BOX_Y = content_box(donor=True)
BOX_PLAIN_X, _ = content_box(donor=False)

# One scale for both panels so the molecule is drawn identically in each.  The
# row spans the full canvas width, and the canvas height then follows from the
# molecule it has to hold, so no band of the figure is left empty.
SPAN_Y = BOX_Y[1] - BOX_Y[0]
SPAN_X = (BOX_DONOR_X[1] - BOX_DONOR_X[0]) + (BOX_PLAIN_X[1] - BOX_PLAIN_X[0])
SCALE = (X_SPAN - 2.0 * MARGIN - GUTTER) / SPAN_X

PANEL_SPAN = (
    (MARGIN, MARGIN + (BOX_DONOR_X[1] - BOX_DONOR_X[0]) * SCALE),
    (X_SPAN - MARGIN - (BOX_PLAIN_X[1] - BOX_PLAIN_X[0]) * SCALE, X_SPAN - MARGIN),
)

DETAIL_Y = 0.31
EQUATION_Y = DETAIL_Y + 0.34
DRAW_BOTTOM = EQUATION_Y + 0.32
DRAW_TOP = DRAW_BOTTOM + SPAN_Y * SCALE
HEADER_Y = DRAW_TOP + 0.33
Y_SPAN = HEADER_Y + 0.34
FIG_HEIGHT = Y_SPAN * FIG_WIDTH / X_SPAN


@dataclass(frozen=True)
class Frame:
    """Maps ideal molecule coordinates onto figure data coordinates."""

    scale: float
    origin: tuple[float, float]

    def at(self, point: tuple[float, float]) -> tuple[float, float]:
        return (
            self.origin[0] + self.scale * point[0],
            self.origin[1] + self.scale * point[1],
        )

    def points(self, length: float) -> float:
        return length * self.scale * PT_PER_UNIT


def build_frame(panel: int) -> Frame:
    """Place a panel's content box flush inside its column of the row."""

    box_x = BOX_DONOR_X if panel == 0 else BOX_PLAIN_X
    return Frame(
        scale=SCALE,
        origin=(
            PANEL_SPAN[panel][0] - SCALE * box_x[0],
            0.5 * (DRAW_TOP + DRAW_BOTTOM) - SCALE * 0.5 * (BOX_Y[0] + BOX_Y[1]),
        ),
    )


def draw_plain_node(ax: plt.Axes, position: tuple[float, float], radius: float) -> None:
    ax.add_patch(
        Circle(
            position,
            radius,
            facecolor=NODE_FILL,
            edgecolor=NODE_EDGE,
            linewidth=0.9,
            zorder=5,
        )
    )


def draw_marked_node(
    ax: plt.Axes,
    position: tuple[float, float],
    radius: float,
    *,
    colour: str,
    label: str,
    stroke: str = "#FFFFFF",
    label_colour: str = "#FFFFFF",
    linewidth: float = 1.0,
) -> None:
    """A filled node carrying the symbol used in the panel caption.

    The default white stroke separates a solid node from the bonds without
    punching a disc out of the ring shading.  The pale donor node instead takes
    a coloured stroke and a coloured symbol, since neither would register on a
    near-white fill.
    """

    ax.add_patch(
        Circle(
            position,
            radius,
            facecolor=colour,
            edgecolor=stroke,
            linewidth=linewidth,
            zorder=5,
        )
    )
    ax.text(
        position[0],
        position[1],
        label,
        ha="center",
        va="center_baseline",
        fontsize=8.6,
        color=label_colour,
        zorder=6,
    )


def draw_molecule(
    ax: plt.Axes,
    frame: Frame,
    *,
    marks: dict[str, tuple[str, str]],
) -> dict[str, tuple[float, float]]:
    positions = {name: frame.at(point) for name, point in ATOMS.items()}

    for ring in RINGS:
        ax.add_patch(
            Polygon(
                [positions[name] for name in ring],
                closed=True,
                facecolor=RING,
                edgecolor="none",
                zorder=0.5,
            )
        )

    for source, target in BONDS:
        x0, y0 = positions[source]
        x1, y1 = positions[target]
        ax.plot(
            [x0, x1],
            [y0, y1],
            color=BOND,
            linewidth=1.45,
            solid_capstyle="round",
            zorder=1,
        )

    for name, position in positions.items():
        if name in marks:
            colour, label = marks[name]
            draw_marked_node(
                ax, position, frame.scale * R_MARK, colour=colour, label=label
            )
        else:
            draw_plain_node(ax, position, frame.scale * R_PLAIN)

    return positions


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    colour: str,
    shrink_a: float,
    shrink_b: float,
    double: bool = False,
    rad: float = 0.0,
    linewidth: float = 1.5,
    mutation: float = 11.0,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="<|-|>" if double else "-|>",
            mutation_scale=mutation,
            linewidth=linewidth,
            color=colour,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=shrink_a,
            shrinkB=shrink_b,
            capstyle="round",
            joinstyle="round",
            zorder=8,
        )
    )


def draw_donor_swap(ax: plt.Axes) -> None:
    frame = build_frame(0)
    positions = draw_molecule(ax, frame, marks={TARGET: (PURPLE, "$v$")})

    target = positions[TARGET]
    donor = frame.at(DONOR)
    draw_marked_node(
        ax,
        donor,
        frame.scale * R_MARK,
        colour=PURPLE_PALE,
        label="$w$",
        stroke=PURPLE,
        label_colour=PURPLE,
        linewidth=1.15,
    )
    ax.text(
        donor[0],
        donor[1] - frame.scale * (R_MARK + 0.30),
        "donor",
        ha="center",
        va="top",
        fontsize=8.0,
        color=PURPLE,
    )

    gap = frame.points(R_MARK * R_GAP) + 2.0
    arrow(ax, donor, target, colour=ARROW, shrink_a=gap, shrink_b=gap)


def draw_node_transposition(ax: plt.Axes) -> None:
    frame = build_frame(1)
    positions = draw_molecule(
        ax,
        frame,
        marks={TARGET: (PURPLE, "$v$"), PARTNER: (PURPLE, "$u$")},
    )

    gap = frame.points(R_MARK * R_GAP) + 2.0
    arrow(
        ax,
        positions[TARGET],
        positions[PARTNER],
        colour=ARROW,
        shrink_a=gap,
        shrink_b=gap,
        double=True,
        rad=SWAP_RAD,
    )


def panel_header(ax: plt.Axes, left: float, letter: str, title: str) -> None:
    ax.text(
        left,
        HEADER_Y,
        letter,
        ha="left",
        va="center",
        fontsize=10.5,
        weight="bold",
        color=INK,
    )
    ax.text(
        left + 0.30,
        HEADER_Y,
        title,
        ha="left",
        va="center",
        fontsize=9.6,
        color=INK,
    )


def caption(
    figure: plt.Figure,
    ax: plt.Axes,
    centre_x: float,
    *,
    equation: tuple[tuple[str, str, float], ...],
    detail: str,
) -> None:
    """Set the equation as coloured fragments so each symbol matches its node."""

    renderer = figure.canvas.get_renderer()
    inverse = ax.transData.inverted()
    pieces = [
        ax.text(
            0.0,
            EQUATION_Y,
            fragment,
            ha="left",
            va="center",
            fontsize=9.8,
            color=colour,
            zorder=3,
        )
        for fragment, colour, _ in equation
    ]

    widths = []
    for piece in pieces:
        box = piece.get_window_extent(renderer=renderer)
        left = inverse.transform((box.x0, box.y0))[0]
        right = inverse.transform((box.x1, box.y0))[0]
        widths.append(right - left)

    gaps = [gap for _, _, gap in equation]
    total = sum(widths) + sum(gaps)
    cursor = centre_x - 0.5 * total
    for piece, width, gap in zip(pieces, widths, gaps):
        cursor += gap
        piece.set_x(cursor)
        cursor += width

    ax.text(
        centre_x,
        DETAIL_Y,
        detail,
        ha="center",
        va="center",
        fontsize=8.2,
        color=MUTED,
        zorder=3,
    )


def make_figure() -> plt.Figure:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": list(FONT_STACK),
            "mathtext.fontset": "custom",
            "mathtext.rm": MATH_FONT,
            "mathtext.it": f"{MATH_FONT}:italic",
            "mathtext.bf": f"{MATH_FONT}:bold",
            "font.size": 8.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    # The axes fills the canvas so that one data unit is exactly
    # FIG_WIDTH / X_SPAN inches, which keeps the arrow gaps (set in points)
    # locked to the node radii.
    figure = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    axis = figure.add_axes((0.0, 0.0, 1.0, 1.0))
    axis.set_xlim(0.0, X_SPAN)
    axis.set_ylim(0.0, Y_SPAN)
    axis.set_aspect("equal")
    axis.axis("off")

    panel_header(axis, PANEL_SPAN[0][0], "a", "Donor swap")
    panel_header(axis, PANEL_SPAN[1][0], "b", "Node transposition")
    draw_donor_swap(axis)
    draw_node_transposition(axis)

    caption(
        figure,
        axis,
        0.5 * (PANEL_SPAN[0][0] + PANEL_SPAN[0][1]),
        equation=(
            ("$x_v$", PURPLE, 0.0),
            (r"$\leftarrow$", INK, 0.045),
            ("$x_w$", PURPLE, 0.045),
        ),
        detail="replace semantic or structural features",
    )
    caption(
        figure,
        axis,
        0.5 * (PANEL_SPAN[1][0] + PANEL_SPAN[1][1]),
        equation=(
            ("$($", INK, 0.0),
            ("$x_u$", PURPLE, -0.015),
            ("$,$", INK, -0.020),
            ("$x_v$", PURPLE, 0.025),
            ("$)$", INK, -0.010),
            (r"$\leftarrow$", INK, 0.055),
            ("$($", INK, 0.055),
            ("$x_v$", PURPLE, -0.015),
            ("$,$", INK, -0.020),
            ("$x_u$", PURPLE, 0.025),
            ("$)$", INK, -0.010),
        ),
        detail="swap semantic or structural features",
    )
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
    figure.savefig(pdf_path, bbox_inches="tight", pad_inches=0.04)
    figure.savefig(png_path, bbox_inches="tight", pad_inches=0.04, dpi=args.dpi)
    plt.close(figure)
    print(pdf_path)
    print(png_path)


if __name__ == "__main__":
    main()
