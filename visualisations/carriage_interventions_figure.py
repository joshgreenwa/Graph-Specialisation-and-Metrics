"""Draw a paper-style schematic of the carriage interventions.

The figure separates the two intervention families implemented in
``graph_specialisation_metrics.carriage``:

* semantic carriage: replace one node content row with a donor row from another
  graph, while structure-derived tensors stay fixed;
* structural carriage: hold content fixed and transpose topology-derived
  structure between an anchor ``u`` and a degree-matched partner ``v``.

The right side is deliberately left at the intervention stage. A later version
can extend the same rows into ``Delta h_i = h_i(clean) - h_i(intervention)``,
``F_sens``, and ``B`` panels.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

_MPLCONFIGDIR = Path(tempfile.gettempdir()) / "graph_specialisation_metrics_matplotlib"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


CONTENT_COLORS = {
    0: "#87b7e5",
    1: "#f2a35e",
    2: "#7bc9a5",
    3: "#e98b96",
    4: "#caa3e8",
    5: "#f0d76f",
    6: "#9cc7cf",
    "donor": "#e45b4f",
}
STRUCT_COLORS = {
    "u": "#59b18b",
    "v": "#7c61c6",
    "j": "#476a9e",
}

NODES = tuple(range(7))
EDGES = (
    (0, 1),
    (1, 2),
    (0, 3),
    (1, 4),
    (2, 4),
    (2, 6),
    (3, 4),
    (4, 5),
    (5, 6),
)
POS = {
    0: (0.00, 1.12),
    1: (0.95, 1.75),
    2: (1.95, 1.18),
    3: (0.20, 0.15),
    4: (1.12, 0.68),
    5: (2.08, 0.18),
    6: (2.88, 1.08),
}

SOURCE_J = 2
ANCHOR_U = 1
PARTNER_V = 4


def add_panel(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    title: str,
    subtitle: str | None = None,
    face: str = "#f7f9fb",
    edge: str = "#c6d2dc",
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.045,rounding_size=0.16",
            facecolor=face,
            edgecolor=edge,
            linewidth=1.0,
            zorder=0,
        )
    )
    ax.text(x + 0.18, y + h - 0.20, title, ha="left", va="top", fontsize=9.8, weight=500)
    if subtitle:
        ax.text(x + 0.18, y + h - 0.44, subtitle, ha="left", va="top", fontsize=7.8, color="#3f4e5a")


def draw_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = "#263744",
    rad: float = 0.0,
    lw: float = 1.25,
    mutation: float = 13,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=mutation,
            linewidth=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=0,
            shrinkB=4,
            zorder=8,
        )
    )


def graph_positions(origin: tuple[float, float], scale: float) -> dict[int, tuple[float, float]]:
    ox, oy = origin
    return {node: (ox + scale * px, oy + scale * py) for node, (px, py) in POS.items()}


def draw_graph(
    ax: plt.Axes,
    origin: tuple[float, float],
    *,
    scale: float = 0.64,
    content_override: dict[int, str] | None = None,
    halos: dict[int, str] | None = None,
    labels: dict[int, str] | None = None,
    changed_nodes: set[int] | None = None,
    structural_edges: tuple[tuple[int, int], ...] = EDGES,
    dim_unhighlighted: bool = False,
) -> dict[int, tuple[float, float]]:
    positions = graph_positions(origin, scale)
    halos = halos or {}
    labels = labels or {}
    content_override = content_override or {}
    changed_nodes = changed_nodes or set()

    for a, b in structural_edges:
        x0, y0 = positions[a]
        x1, y1 = positions[b]
        ax.plot([x0, x1], [y0, y1], color="#20262d", lw=1.05, zorder=1, solid_capstyle="round")

    r = 0.145 * scale / 0.76
    for node in NODES:
        x, y = positions[node]
        if node in halos:
            ax.add_patch(
                Circle(
                    (x, y),
                    r * 1.60,
                    facecolor=halos[node],
                    edgecolor="none",
                    alpha=0.22,
                    zorder=2,
                )
            )
            ax.add_patch(
                Circle(
                    (x, y),
                    r * 1.43,
                    facecolor="none",
                    edgecolor=halos[node],
                    linewidth=1.25,
                    zorder=4,
                )
            )
        fill = content_override.get(node, CONTENT_COLORS[node])
        alpha = 0.42 if dim_unhighlighted and node not in changed_nodes and node not in halos else 1.0
        ax.add_patch(
            Circle(
                (x, y),
                r,
                facecolor=fill,
                edgecolor="#111111",
                linewidth=0.95,
                alpha=alpha,
                zorder=5,
            )
        )
        if node in changed_nodes:
            ax.add_patch(
                Circle((x, y), r * 1.25, facecolor="none", edgecolor="#111111", linewidth=1.15, zorder=6)
            )
        if node in labels:
            ax.text(x, y, labels[node], ha="center", va="center", fontsize=8.5, color="#101418", zorder=7)
    return positions


def draw_tag(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    *,
    face: str,
    edge: str,
    text_color: str = "#1a242c",
    width: float | None = None,
) -> None:
    w = width if width is not None else 0.13 * len(text) + 0.42
    h = 0.34
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.03,rounding_size=0.09",
            facecolor=face,
            edgecolor=edge,
            linewidth=0.85,
            zorder=9,
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.3, color=text_color, zorder=10)


def draw_semantic_operation(ax: plt.Axes, x: float, y: float) -> None:
    add_panel(
        ax,
        x,
        y,
        2.25,
        2.18,
        title="Donor content swap",
        subtitle=r"$x_j \leftarrow \tilde{x}$,  with $S$ fixed",
        face="#fff7f5",
        edge="#efc7bd",
    )
    donor_x = x + 0.62
    donor_y = y + 0.88
    ax.add_patch(Circle((donor_x, donor_y), 0.22, facecolor=CONTENT_COLORS["donor"], edgecolor="#111111", lw=1.0, zorder=4))
    ax.text(donor_x, donor_y - 0.01, r"$\tilde{x}$", ha="center", va="center", fontsize=10.5, zorder=5)
    ax.text(donor_x, donor_y + 0.38, "donor row", ha="center", va="bottom", fontsize=7.8, color="#4a2c27")
    draw_arrow(ax, (donor_x + 0.32, donor_y), (x + 1.72, donor_y), color="#b54136", lw=1.2)
    ax.text(x + 1.72, donor_y, r"$j$", ha="center", va="center", fontsize=10.5)
    ax.add_patch(Rectangle((x + 1.54, donor_y - 0.18), 0.36, 0.36, facecolor="none", edgecolor="#b54136", lw=1.2))
    ax.text(x + 1.13, y + 0.30, r"average over $K$ donors", ha="center", va="center", fontsize=7.8, color="#6c3932")


def draw_structural_operation(ax: plt.Axes, x: float, y: float) -> None:
    add_panel(
        ax,
        x,
        y,
        2.25,
        2.18,
        title="Structural transposition",
        subtitle=r"$S' = P_{(uv)} S P_{(uv)}^\top$,  with $X$ fixed",
        face="#f4fbf7",
        edge="#b8dec9",
    )
    ux, uy = x + 0.66, y + 0.92
    vx, vy = x + 1.62, y + 0.92
    ax.add_patch(Circle((ux, uy), 0.26, facecolor=STRUCT_COLORS["u"], edgecolor="#111111", lw=1.1, alpha=0.38, zorder=3))
    ax.add_patch(Circle((vx, vy), 0.26, facecolor=STRUCT_COLORS["v"], edgecolor="#111111", lw=1.1, alpha=0.38, zorder=3))
    ax.text(ux, uy, r"$s_u$", ha="center", va="center", fontsize=10.5, zorder=4)
    ax.text(vx, vy, r"$s_v$", ha="center", va="center", fontsize=10.5, zorder=4)
    ax.add_patch(
        FancyArrowPatch(
            (ux + 0.30, uy + 0.10),
            (vx - 0.30, vy + 0.10),
            arrowstyle="<->",
            mutation_scale=12,
            linewidth=1.2,
            color="#2e6b4d",
            connectionstyle="arc3,rad=0.30",
            zorder=5,
        )
    )
    ax.text(x + 1.13, y + 0.30, r"average over $K$ degree-matched partners", ha="center", va="center", fontsize=7.3, color="#315d43")


def make_figure() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(13.2, 6.1))
    ax.set_xlim(0, 12.8)
    ax.set_ylim(0, 6.0)
    ax.axis("off")

    ax.text(0.30, 5.78, "A", fontsize=12.5, weight=600, ha="left", va="top")
    ax.text(0.62, 5.78, "Semantic intervention", fontsize=12.5, weight=500, ha="left", va="top", color="#8f2e27")
    ax.text(0.62, 5.48, "Change X only; keep S fixed.", fontsize=8.6, ha="left", va="top", color="#52606d")

    ax.text(0.30, 2.78, "B", fontsize=12.5, weight=600, ha="left", va="top")
    ax.text(0.62, 2.78, "Structural intervention", fontsize=12.5, weight=500, ha="left", va="top", color="#2f6849")
    ax.text(0.62, 2.48, "Change S only; keep X fixed.", fontsize=8.6, ha="left", va="top", color="#52606d")

    # Row A: semantic.
    add_panel(
        ax,
        0.42,
        3.22,
        3.20,
        2.00,
        title="Clean graph",
        face="#f8fafc",
        edge="#c7d1db",
    )
    draw_graph(
        ax,
        (0.88, 3.47),
        halos={SOURCE_J: STRUCT_COLORS["j"]},
        labels={SOURCE_J: r"$j$"},
        changed_nodes={SOURCE_J},
    )
    draw_tag(ax, 2.55, 3.36, "source j", face="#e8eef8", edge="#b7c6dc", width=0.78)

    draw_semantic_operation(ax, 4.03, 3.22)

    add_panel(
        ax,
        6.75,
        3.22,
        3.30,
        2.00,
        title="Intervened graph",
        face="#f8fafc",
        edge="#c7d1db",
    )
    draw_graph(
        ax,
        (7.22, 3.47),
        content_override={SOURCE_J: CONTENT_COLORS["donor"]},
        halos={SOURCE_J: STRUCT_COLORS["j"]},
        labels={SOURCE_J: r"$j$"},
        changed_nodes={SOURCE_J},
    )
    draw_tag(ax, 9.08, 3.36, "S fixed", face="#edf5fb", edge="#bcd0df", width=0.76)

    draw_arrow(ax, (3.68, 4.20), (4.00, 4.20), color="#6e7f8c")
    draw_arrow(ax, (6.32, 4.20), (6.72, 4.20), color="#6e7f8c")

    # Row B: structural.
    add_panel(
        ax,
        0.42,
        0.35,
        3.20,
        2.00,
        title="Clean graph",
        face="#f8fafc",
        edge="#c7d1db",
    )
    draw_graph(
        ax,
        (0.88, 0.60),
        halos={ANCHOR_U: STRUCT_COLORS["u"], PARTNER_V: STRUCT_COLORS["v"]},
        labels={ANCHOR_U: r"$u$", PARTNER_V: r"$v$"},
        changed_nodes={ANCHOR_U, PARTNER_V},
    )
    draw_tag(ax, 2.42, 0.47, "roles u, v", face="#edf7f1", edge="#bdd8c8", width=0.90)

    draw_structural_operation(ax, 4.03, 0.35)

    add_panel(
        ax,
        6.75,
        0.35,
        3.30,
        2.00,
        title="Intervened graph",
        face="#f8fafc",
        edge="#c7d1db",
    )
    draw_graph(
        ax,
        (7.22, 0.60),
        halos={ANCHOR_U: STRUCT_COLORS["v"], PARTNER_V: STRUCT_COLORS["u"]},
        labels={ANCHOR_U: r"$u$", PARTNER_V: r"$v$"},
        changed_nodes={ANCHOR_U, PARTNER_V},
    )
    draw_tag(ax, 9.08, 0.47, "X fixed", face="#f7f1ff", edge="#cabbe0", width=0.76)

    draw_arrow(ax, (3.68, 1.33), (4.00, 1.33), color="#6e7f8c")
    draw_arrow(ax, (6.32, 1.33), (6.72, 1.33), color="#6e7f8c")

    # Legend.
    lx, ly = 10.58, 4.80
    ax.text(lx, ly, "Visual key", ha="left", va="center", fontsize=10.8, weight=500)
    ax.add_patch(Circle((lx + 0.16, ly - 0.48), 0.10, facecolor=CONTENT_COLORS[1], edgecolor="#111111", lw=1.0))
    ax.text(lx + 0.38, ly - 0.48, r"inner fill: content row $x_i$", ha="left", va="center", fontsize=8.6, color="#35424d")
    ax.add_patch(Circle((lx + 0.16, ly - 0.92), 0.16, facecolor=STRUCT_COLORS["u"], edgecolor="none", alpha=0.22))
    ax.add_patch(Circle((lx + 0.16, ly - 0.92), 0.14, facecolor="none", edgecolor=STRUCT_COLORS["u"], lw=1.3))
    ax.text(lx + 0.38, ly - 0.92, r"halo: structural role / support $s_i$", ha="left", va="center", fontsize=8.6, color="#35424d")

    ax.text(
        10.55,
        2.12,
        "Both rows then feed the\nsame carriage estimator:\n"
        r"$\Delta h_i = h_i^L(\mathrm{clean}) - h_i^L(\mathrm{intervention})$",
        ha="left",
        va="top",
        fontsize=9.0,
        color="#3f4e5a",
        linespacing=1.35,
    )

    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("visualisations/generated"),
        help="Directory where the PDF and PNG will be written.",
    )
    parser.add_argument("--stem", default="carriage_interventions", help="Output filename stem.")
    parser.add_argument("--dpi", type=int, default=300, help="PNG export DPI.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig = make_figure()
    pdf_path = args.out_dir / f"{args.stem}.pdf"
    png_path = args.out_dir / f"{args.stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.05, dpi=args.dpi)
    plt.close(fig)
    print(pdf_path)
    print(png_path)


if __name__ == "__main__":
    main()
