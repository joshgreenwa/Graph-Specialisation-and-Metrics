"""Focused publication plots shared by the ZINC and QM9 GRIT analyses."""

from __future__ import annotations

from collections import Counter
import io
import json
import os
from pathlib import Path
import tempfile
import textwrap
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

from .grit_figure_data import CanonicalHeadMetrics, Head, figure_identity


NAVY = "#17324D"
BLUE = "#2878B5"
TEAL = "#087E8B"
GOLD = "#E6A700"
ORANGE = "#D97706"
SLATE = "#607080"
LIGHT_GRID = "#DCE3E8"
ATTENTION_CMAP = plt.get_cmap("Blues")
SELECTIVITY_CMAP = plt.get_cmap("coolwarm")
PUBLICATION_PNG_DPI = 600
PUBLICATION_PDF_RASTER_DPI = 1200
MOLECULE_RENDER_DPI = 600
# Backwards-compatible name retained for existing figure metadata consumers.
MOLECULE_DRAW_DPI = MOLECULE_RENDER_DPI
CORE_SCATTER_FIGSIZE = (8.0, 5.9)
HEAD_STYLES = {
    "semantic": {"color": GOLD, "label": "Semantic specialist"},
    "structural": {"color": TEAL, "label": "Structural specialist"},
    "structural_alternate": {
        "color": "#2A9D8F",
        "label": "Alternate structural specialist",
    },
}
# Stable exact chemistry-focus identities shared with the Graphormer PCQM
# figures.  A label never receives a plot-local or frequency-dependent colour.
PCA_FOCUS_COLORS = {
    "Ring: junction": "#17324D",
    "Ring: aromatic": "#2878B5",
    "Ring: aliphatic": "#56A9D8",
    "O: carbonyl": "#C45100",
    "O: ester/carboxyl": "#E17C05",
    "O: hydroxyl": "#F2B134",
    "O: other": "#A65D26",
    "N: aromatic": "#006D5B",
    "N: nitrile": "#008E72",
    "N: nitro": "#2AA876",
    "N: amide": "#57B894",
    "N: other": "#7EC8AE",
    "X: halogen": "#7B61A8",
    "S: sulfur": "#CCB000",
    "P: phosphorus": "#8B6F47",
    "Branch: degree>=3": "#6B8E23",
    "Charge: +": "#C44E52",
    "Charge: -": "#9C4F96",
    "H: hydrogen": "#8C9AA5",
    "other/diffuse": "#B8C2CA",
    "Other / rare": "#4B5563",
}

# Backwards-compatible public name; values now follow the PCQM chemical-group
# convention rather than element/type-ID labels.
ELEMENT_COLORS = PCA_FOCUS_COLORS


def apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": PUBLICATION_PNG_DPI,
            "savefig.transparent": False,
            "savefig.pad_inches": 0.04,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
            "mathtext.fontset": "dejavusans",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.edgecolor": NAVY,
            "axes.linewidth": 0.9,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "grid.color": LIGHT_GRID,
            "grid.linewidth": 0.7,
            "grid.alpha": 0.75,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "pdf.compression": 9,
            "pdf.use14corefonts": False,
            "ps.fonttype": 42,
            "path.simplify": False,
        }
    )


def _head_label(head: Head) -> str:
    return f"L{head[0]} H{head[1]}"


def _heatmap(ax, values, *, title: str, cmap, norm):
    image = ax.imshow(
        values,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
        rasterized=True,
    )
    layers, heads = values.shape
    ax.set_title(title, pad=10)
    ax.set_xlabel("Head index")
    ax.set_ylabel("Layer index")
    ax.set_xticks(np.arange(heads))
    ax.set_xticklabels(np.arange(heads), fontsize=6.5)
    ax.set_yticks(np.arange(layers))
    ax.set_yticklabels(np.arange(layers))
    ax.tick_params(length=0)
    return image


def plot_score_heatmaps(
    metrics: CanonicalHeadMetrics,
    selected_heads: Mapping[str, Head] | None = None,
    *,
    title: str,
):
    del selected_heads
    apply_publication_style()
    values = np.concatenate(
        (
            metrics.normalized_semantic.reshape(-1),
            metrics.normalized_structural.reshape(-1),
        )
    )
    vmax = max(float(np.nanpercentile(values, 99.5)), 1.0)
    norm = Normalize(vmin=0.0, vmax=vmax)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14.0, 5.2),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    semantic = _heatmap(
        axes[0],
        metrics.normalized_semantic,
        title=r"Semantic score $S_{\rm sem}/\overline{S}_{\rm sem}$",
        cmap="viridis",
        norm=norm,
    )
    _heatmap(
        axes[1],
        metrics.normalized_structural,
        title=r"Structural score $S_{\rm str}/\overline{S}_{\rm str}$",
        cmap="viridis",
        norm=norm,
    )
    colorbar = fig.colorbar(semantic, ax=axes, shrink=0.86, pad=0.015)
    colorbar.set_label("Normalised score")
    fig.suptitle(title, fontsize=18, y=1.04)
    return fig


def plot_coordinate_heatmaps(
    metrics: CanonicalHeadMetrics,
    selected_heads: Mapping[str, Head] | None = None,
    *,
    title: str,
):
    del selected_heads
    apply_publication_style()
    active_values = metrics.selectivity[
        metrics.active & np.isfinite(metrics.selectivity)
    ]
    d_max = max(
        0.1,
        min(
            1.0,
            float(np.max(np.abs(active_values))) if active_values.size else 0.1,
        ),
    )
    finite_joint = metrics.joint_sensitivity[
        np.isfinite(metrics.joint_sensitivity)
    ]
    j_max = max(
        1.0,
        float(np.percentile(finite_joint, 99.5)) if finite_joint.size else 1.0,
    )
    selectivity_cmap = SELECTIVITY_CMAP.copy()
    selectivity_cmap.set_bad("#D3D9DE")
    active_selectivity = np.ma.array(metrics.selectivity, mask=~metrics.active)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14.0, 5.2),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    selectivity = _heatmap(
        axes[0],
        active_selectivity,
        title=r"Relative selectivity $D_{\rm rel}$ (active heads)",
        cmap=selectivity_cmap,
        norm=TwoSlopeNorm(vmin=-d_max, vcenter=0.0, vmax=d_max),
    )
    joint = _heatmap(
        axes[1],
        metrics.joint_sensitivity,
        title=r"Joint sensitivity $J$",
        cmap="viridis",
        norm=Normalize(vmin=0.0, vmax=j_max),
    )
    d_bar = fig.colorbar(selectivity, ax=axes[0], shrink=0.86, pad=0.015)
    d_bar.set_label(
        r"$D_{\rm rel}$  (structural $\leftarrow$  $\rightarrow$ semantic)"
    )
    j_bar = fig.colorbar(joint, ax=axes[1], shrink=0.86, pad=0.015)
    j_bar.set_label(r"$J$")
    fig.suptitle(title, fontsize=18, y=1.04)
    return fig


def _scatter_heads(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    *,
    active: np.ndarray | None = None,
):
    layers, heads = x.shape
    layer_ids = np.repeat(np.arange(layers), heads)
    finite = np.isfinite(x.reshape(-1)) & np.isfinite(y.reshape(-1))
    if active is not None:
        inactive = finite & ~active.reshape(-1)
        if np.any(inactive):
            ax.scatter(
                x.reshape(-1)[inactive],
                y.reshape(-1)[inactive],
                s=24,
                color="#C7D0D6",
                alpha=0.55,
                linewidth=0,
                label="Inactive",
                zorder=1,
            )
        finite &= active.reshape(-1)
    return ax.scatter(
        x.reshape(-1)[finite],
        y.reshape(-1)[finite],
        c=layer_ids[finite],
        cmap="viridis",
        norm=Normalize(0, max(layers - 1, 1)),
        s=42,
        edgecolor="white",
        linewidth=0.45,
        alpha=0.92,
        zorder=2,
    )


def _annotate_selected_scatter(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    selected_heads: Mapping[str, Head] | None,
) -> None:
    if not selected_heads:
        return
    for role, head in selected_heads.items():
        layer, index = head
        style = HEAD_STYLES.get(role, {"color": NAVY, "label": role.title()})
        point_x, point_y = float(x[layer, index]), float(y[layer, index])
        x_limits, y_limits = ax.get_xlim(), ax.get_ylim()
        x_fraction = (point_x - x_limits[0]) / max(
            x_limits[1] - x_limits[0], 1e-12
        )
        y_fraction = (point_y - y_limits[0]) / max(
            y_limits[1] - y_limits[0], 1e-12
        )
        x_offset = -10 if x_fraction > 0.72 else 10
        y_offset = -14 if y_fraction > 0.76 else 10
        ax.scatter(
            [point_x],
            [point_y],
            s=110,
            facecolors="none",
            edgecolors=style["color"],
            linewidths=2.1,
            zorder=5,
        )
        ax.annotate(
            f"{style['label']} ({_head_label(head)})",
            xy=(point_x, point_y),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            color=style["color"],
            fontsize=9,
            fontweight="bold",
            ha="right" if x_offset < 0 else "left",
            va="top" if y_offset < 0 else "bottom",
            bbox={
                "boxstyle": "round,pad=0.28",
                "facecolor": "white",
                "edgecolor": style["color"],
                "linewidth": 0.8,
                "alpha": 0.94,
            },
            arrowprops={
                "arrowstyle": "-",
                "color": style["color"],
                "linewidth": 1.1,
            },
            zorder=6,
        )


def plot_score_plane(
    metrics: CanonicalHeadMetrics,
    selected_heads: Mapping[str, Head] | None = None,
    *,
    title: str,
):
    apply_publication_style()
    x, y = metrics.normalized_structural, metrics.normalized_semantic
    fig, ax = plt.subplots(
        figsize=CORE_SCATTER_FIGSIZE, constrained_layout=True
    )
    scatter = _scatter_heads(ax, x, y)
    maximum = max(float(np.nanmax(x)), float(np.nanmax(y))) * 1.06
    ax.plot([0, maximum], [0, maximum], color=SLATE, linestyle="--", linewidth=1.2)
    ax.set_xlim(0, maximum)
    ax.set_ylim(0, maximum)
    ax.set_xlabel(r"Structural score $S_{\rm str}/\overline{S}_{\rm str}$")
    ax.set_ylabel(r"Semantic score $S_{\rm sem}/\overline{S}_{\rm sem}$")
    ax.set_title(title, fontsize=17, pad=12)
    ax.grid(True)
    _annotate_selected_scatter(ax, x, y, selected_heads)
    colorbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    colorbar.set_label("Layer")
    colorbar.set_ticks(np.arange(metrics.num_layers))
    return fig


def automatic_selectivity_limits(
    selectivity: np.ndarray,
    active: np.ndarray | None = None,
) -> tuple[float, float]:
    finite = np.isfinite(selectivity)
    if active is not None:
        finite &= active
    values = selectivity[finite]
    if not values.size:
        return -0.1, 0.1
    lower, upper = np.nanpercentile(values, [0.25, 99.75])
    span = max(float(upper - lower), 0.1)
    lower = min(float(lower - 0.08 * span), -0.03)
    upper = max(float(upper + 0.08 * span), 0.03)
    return max(-1.0, lower), min(1.0, upper)


def plot_selectivity_joint_plane(
    metrics: CanonicalHeadMetrics,
    selected_heads: Mapping[str, Head] | None = None,
    *,
    xlim: tuple[float, float] | None = None,
    active_only: bool = True,
    title: str,
):
    apply_publication_style()
    x, y = metrics.selectivity, metrics.joint_sensitivity
    fig, ax = plt.subplots(
        figsize=CORE_SCATTER_FIGSIZE, constrained_layout=True
    )
    scatter = _scatter_heads(
        ax, x, y, active=metrics.active if active_only else None
    )
    ax.axvline(0, color=SLATE, linestyle="--", linewidth=1.1)
    ax.set_xlim(
        *(xlim or automatic_selectivity_limits(x, metrics.active if active_only else None))
    )
    ax.set_ylim(bottom=0, top=float(np.nanmax(y)) * 1.08)
    ax.set_xlabel(
        r"Relative selectivity $D_{\rm rel}$"
        "\n(structural $\\leftarrow$   $\\rightarrow$ semantic)"
    )
    ax.set_ylabel(r"Joint sensitivity $J$")
    ax.set_title(title, fontsize=17, pad=12)
    ax.grid(True)
    _annotate_selected_scatter(ax, x, y, selected_heads)
    if active_only and np.any(~metrics.active):
        ax.legend(loc="upper left", fontsize=9)
    colorbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    colorbar.set_label("Layer")
    colorbar.set_ticks(np.arange(metrics.num_layers))
    return fig


def _node_conditioned_attention(attention: np.ndarray) -> np.ndarray:
    attention = np.asarray(attention, dtype=np.float64)
    if attention.ndim != 2 or attention.shape[0] != attention.shape[1]:
        raise ValueError(f"expected square attention matrix, got {attention.shape}")
    denominator = np.clip(attention.sum(axis=-1, keepdims=True), 1e-12, None)
    return attention / denominator


def _molecule_from_example(example: Mapping[str, Any]):
    from rdkit import Chem

    mol_block = example.get("mol_block")
    molecule = (
        Chem.MolFromMolBlock(str(mol_block), removeHs=False, sanitize=True)
        if mol_block
        else Chem.MolFromSmiles(str(example["smiles"]))
    )
    if molecule is None:
        raise ValueError("RDKit could not parse the cached molecule")
    expected = int(example.get("n_atoms", molecule.GetNumAtoms()))
    if molecule.GetNumAtoms() != expected:
        raise ValueError(
            f"cached molecule has {molecule.GetNumAtoms()} atoms; expected {expected}"
        )
    return molecule


def _prepare_molecule(example: Mapping[str, Any]):
    from rdkit.Chem import rdDepictor

    molecule = _molecule_from_example(example)
    rdDepictor.Compute2DCoords(molecule)
    return molecule


def _draw_molecule_plain(
    example: Mapping[str, Any],
    *,
    figsize: tuple[float, float] = (4.2, 3.7),
    dpi: int = MOLECULE_RENDER_DPI,
):
    from PIL import Image
    from rdkit.Chem.Draw import rdMolDraw2D

    molecule = _prepare_molecule(example)
    width, height = int(figsize[0] * dpi), int(figsize[1] * dpi)
    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    options = drawer.drawOptions()
    options.addAtomIndices = False
    options.bondLineWidth = 5.0
    options.fixedFontSize = 44
    options.padding = 0.06
    for index in range(molecule.GetNumAtoms()):
        options.atomLabels[index] = str(index)
    drawer.DrawMolecule(molecule, legend="")
    drawer.FinishDrawing()
    return Image.open(io.BytesIO(drawer.GetDrawingText()))


def _draw_molecule_attention(
    example: Mapping[str, Any],
    *,
    inbound: np.ndarray,
    vmax: float,
    figsize: tuple[float, float] = (4.2, 3.7),
    dpi: int = MOLECULE_RENDER_DPI,
):
    """RDKit molecule with atom-centred attention-inflow highlights."""

    from PIL import Image
    from rdkit.Chem.Draw import rdMolDraw2D

    molecule = _prepare_molecule(example)
    inbound = np.asarray(inbound, dtype=np.float64)
    if molecule.GetNumAtoms() != len(inbound):
        raise ValueError(
            f"RDKit has {molecule.GetNumAtoms()} atoms but attention has "
            f"{len(inbound)}"
        )
    norm = Normalize(vmin=0.0, vmax=max(float(vmax), 1e-12), clip=True)
    highlight_atoms = list(range(molecule.GetNumAtoms()))
    highlight_colors = {}
    highlight_radii = {}
    for atom, value in enumerate(inbound):
        scaled = float(norm(value))
        rgba = ATTENTION_CMAP(0.18 + 0.72 * scaled)
        highlight_colors[atom] = tuple(float(channel) for channel in rgba[:3])
        highlight_radii[atom] = 0.20 + 0.30 * np.sqrt(scaled)

    width, height = int(figsize[0] * dpi), int(figsize[1] * dpi)
    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    options = drawer.drawOptions()
    options.addAtomIndices = False
    options.fillHighlights = True
    options.atomHighlightsAreCircles = True
    options.bondLineWidth = 5.0
    options.fixedFontSize = 44
    options.padding = 0.06
    for index in highlight_atoms:
        options.atomLabels[index] = str(index)
    drawer.DrawMolecule(
        molecule,
        legend="",
        highlightAtoms=highlight_atoms,
        highlightAtomColors=highlight_colors,
        highlightAtomRadii=highlight_radii,
    )
    drawer.FinishDrawing()
    return Image.open(io.BytesIO(drawer.GetDrawingText()))


def _payload_display_title(payload: Mapping[str, Any]) -> str:
    if payload.get("display_title"):
        return str(payload["display_title"])
    return figure_identity(str(payload.get("task", "GRIT")))["display_title"]


def _molecule_caption(
    example: Mapping[str, Any],
    *,
    dataset_label: str,
) -> str:
    graph_index = int(example["dataset_index"])
    name = str(example.get("molecule_name") or "").strip()
    identifier = f"{dataset_label} eval {graph_index}"
    if name:
        identifier += f" · {name}"
    formula = str(example.get("formula") or "").strip()
    if formula:
        identifier += f" · {formula}"
    smiles = str(example.get("smiles") or "").strip()
    if smiles:
        identifier += "\n" + textwrap.fill(
            f"canonical SMILES: {smiles}",
            width=52,
            subsequent_indent="  ",
        )
    return identifier


def plot_attention_grid(
    examples_payload: Mapping[str, Any],
    *,
    role: str,
    head: Head,
    per_graph_coordinates: Mapping[int, Mapping[str, float]],
    net_d_rel: float,
    net_joint_sensitivity: float,
    title_label: str | None = None,
):
    """GRIT attention views with aggregate and graph-local coordinates."""

    apply_publication_style()
    examples = list(examples_payload["examples"])
    if not examples:
        raise ValueError("the attention grid requires at least one example")
    matrices = [
        _node_conditioned_attention(example["attention"][role])
        for example in examples
    ]
    matrix_max = max(
        max(float(np.nanpercentile(matrix, 99)), 1e-6) for matrix in matrices
    )
    inbound = [matrix.mean(axis=0) for matrix in matrices]
    inbound_max = max(
        max(float(np.nanpercentile(values, 99)), 1e-6) for values in inbound
    )
    num_rows = len(examples)
    fig = plt.figure(
        figsize=(13.2, 1.65 + 3.35 * num_rows + 0.72),
        constrained_layout=True,
    )
    grid = fig.add_gridspec(
        num_rows + 2,
        3,
        height_ratios=[0.16, *([1.0] * num_rows), 0.11],
        width_ratios=[1.0, 1.08, 1.12],
    )
    title_axis = fig.add_subplot(grid[0, :])
    title_axis.axis("off")
    axes = np.asarray(
        [
            [fig.add_subplot(grid[row + 1, column]) for column in range(3)]
            for row in range(num_rows)
        ],
        dtype=object,
    )
    colorbar_axis = fig.add_subplot(grid[-1, :])
    task = str(examples_payload["task"])
    dataset_label = str(
        examples_payload.get(
            "dataset_label", figure_identity(task)["dataset_label"]
        )
    )
    for row, (example, matrix) in enumerate(zip(examples, matrices)):
        graph_index = int(example["dataset_index"])
        graph_coordinates = per_graph_coordinates.get(
            graph_index, per_graph_coordinates.get(str(graph_index))
        )
        if graph_coordinates is None:
            raise KeyError(
                f"no graph-local coordinates were supplied for eval index {graph_index}"
            )
        d_rel = float(
            graph_coordinates.get(
                "D_rel", graph_coordinates.get("selectivity", np.nan)
            )
        )
        joint = float(
            graph_coordinates.get(
                "J", graph_coordinates.get("joint_sensitivity", np.nan)
            )
        )
        axes[row, 0].imshow(_draw_molecule_plain(example))
        axes[row, 0].axis("off")
        axes[row, 0].text(
            0.01,
            0.99,
            _molecule_caption(example, dataset_label=dataset_label)
            + "\n"
            + rf"Graph-local: $D_{{\rm rel}} = {d_rel:+.3f};\ J = {joint:.3f}$",
            transform=axes[row, 0].transAxes,
            ha="left",
            va="top",
            fontsize=10.5,
            color=NAVY,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.90},
        )
        axes[row, 1].imshow(
            _draw_molecule_attention(
                example,
                inbound=inbound[row],
                vmax=inbound_max,
            )
        )
        axes[row, 1].axis("off")
        image = axes[row, 2].imshow(
            matrix,
            cmap=ATTENTION_CMAP,
            vmin=0,
            vmax=matrix_max,
            interpolation="nearest",
            aspect="equal",
            rasterized=True,
        )
        axes[row, 2].set_xlabel("Key atom", fontsize=13)
        axes[row, 2].set_ylabel("Query atom", fontsize=13)
        axes[row, 2].set_xticks(np.arange(matrix.shape[0]))
        axes[row, 2].set_yticks(np.arange(matrix.shape[0]))
        axes[row, 2].tick_params(labelsize=8, length=2.5)
    for column, label in enumerate(
        ["Molecule", "Attention-weighted molecule", "Node-conditioned attention"]
    ):
        axes[0, column].set_title(label, fontsize=16, pad=10)
    style = HEAD_STYLES.get(role, {"label": role.replace("_", " ").title()})
    title_axis.text(
        0.5,
        0.76,
        f"{_payload_display_title(examples_payload)} — "
        f"{title_label or style['label']} — {_head_label(head)}",
        ha="center",
        va="center",
        fontsize=20,
        color=NAVY,
    )
    title_axis.text(
        0.5,
        0.16,
        rf"Net: $D_{{\rm rel}} = {float(net_d_rel):+.3f};\quad "
        rf"J = {float(net_joint_sensitivity):.3f}$",
        ha="center",
        va="center",
        fontsize=16,
        color=NAVY,
    )
    colorbar = fig.colorbar(
        image, cax=colorbar_axis, orientation="horizontal"
    )
    colorbar.set_label("Attention weight", fontsize=14, labelpad=7)
    colorbar.ax.tick_params(labelsize=11, length=3)
    for tick_label in colorbar.ax.get_xticklabels():
        tick_label.set_fontweight("medium")

    fig.canvas.draw()
    colorbar_position = colorbar_axis.get_position()
    fig.set_layout_engine("none")
    colorbar_axis.set_position(
        [
            colorbar_position.x0 + 0.08 * colorbar_position.width,
            colorbar_position.y0,
            0.84 * colorbar_position.width,
            colorbar_position.height,
        ]
    )
    return fig


def _pca(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vectors = np.asarray(vectors, dtype=np.float64)
    centered = vectors - vectors.mean(axis=0, keepdims=True)
    if np.linalg.norm(centered) < 1e-12:
        return np.zeros((len(vectors), 2)), np.zeros(2)
    _, singular, right = np.linalg.svd(centered, full_matrices=False)
    components = min(2, len(right))
    coordinates = np.zeros((len(vectors), 2), dtype=np.float64)
    coordinates[:, :components] = centered @ right[:components].T
    variance = singular**2
    explained = np.zeros(2, dtype=np.float64)
    explained[:components] = variance[:components] / np.clip(
        variance.sum(), 1e-12, None
    )
    return coordinates, explained


def _group_pca_labels(
    labels: list[str], *, maximum_categories: int = 9, minimum_count: int = 5
) -> list[str]:
    counts = Counter(labels)
    retained = {
        label
        for label, count in counts.most_common(maximum_categories - 1)
        if count >= minimum_count
    }
    return [label if label in retained else "Other / rare" for label in labels]


def _pca_focus_color(label: str) -> str:
    """Return the invariant publication colour for a chemistry-focus label."""

    return PCA_FOCUS_COLORS.get(str(label), SLATE)


def _ordered_pca_categories(labels: Sequence[str]) -> list[str]:
    """Order observed focus labels by count, with deterministic ties."""

    counts = Counter(str(label) for label in labels)
    palette_order = {
        label: position for position, label in enumerate(PCA_FOCUS_COLORS)
    }
    return sorted(
        counts,
        key=lambda label: (
            -counts[label],
            palette_order.get(label, len(palette_order)),
            label.casefold(),
            label,
        ),
    )


def plot_av_pca(
    payload: Mapping[str, Any],
    *,
    maximum_categories: int = 20,
    minimum_count: int = 1,
    title_label: str | None = None,
    d_rel: float | None = None,
    joint_sensitivity: float | None = None,
):
    apply_publication_style()
    coordinates, explained = _pca(np.asarray(payload["vectors"]))
    labels = _group_pca_labels(
        list(payload["labels"]),
        maximum_categories=maximum_categories,
        minimum_count=minimum_count,
    )
    categories = _ordered_pca_categories(labels)
    fig, ax = plt.subplots(figsize=(8.8, 6.2), constrained_layout=True)
    labels_array = np.asarray(labels)
    for category in categories:
        color = _pca_focus_color(category)
        mask = labels_array == category
        ax.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=34,
            color=color,
            edgecolor="white",
            linewidth=0.35,
            alpha=0.78,
            label=f"{category} (n={int(mask.sum())})",
        )
    ax.axhline(0, color=LIGHT_GRID, linewidth=0.8, zorder=0)
    ax.axvline(0, color=LIGHT_GRID, linewidth=0.8, zorder=0)
    ax.set_xlabel(f"PC1 ({100 * explained[0]:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({100 * explained[1]:.1f}% variance)")
    head = tuple(payload["head"])
    metric_line = ""
    if d_rel is not None and joint_sensitivity is not None:
        metric_line = (
            "\n"
            + rf"$D_{{\rm rel}} = {float(d_rel):+.3f};\quad "
            + rf"J = {float(joint_sensitivity):.3f}$"
        )
    descriptor = (
        f"{title_label} — {_head_label(head)}"
        if title_label
        else _head_label(head)
    )
    ax.set_title(
        f"{_payload_display_title(payload)}\n"
        f"PCA of routed head output — {descriptor}"
        f"{metric_line}\n"
        f"$n = {int(payload['n_used'])}$ molecules",
        fontsize=15,
    )
    ax.grid(False)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=9.5,
        markerscale=1.15,
        handletextpad=0.55,
    )
    return fig


def _layer_pca_grid_shape(num_heads: int) -> tuple[int, int]:
    """Choose the closest practical 16:9 panel grid for one GRIT layer."""

    num_heads = int(num_heads)
    if num_heads < 1:
        raise ValueError("a layer PCA grid requires at least one head")
    ncols = min(8, max(1, int(np.ceil(np.sqrt(num_heads * 16 / 9)))))
    nrows = int(np.ceil(num_heads / ncols))
    return nrows, ncols


def plot_layer_av_pca_grid(
    payload: Mapping[str, Any],
    *,
    nrows: int | None = None,
    ncols: int | None = None,
):
    """Plot independently fitted routed-output PCA panels for one layer."""

    apply_publication_style()
    layer = int(payload["layer"])
    vectors = np.asarray(payload["vectors"], dtype=np.float64)
    labels = np.asarray(payload["labels"], dtype=object)
    if vectors.ndim != 3:
        raise ValueError(
            "layer PCA vectors must have shape [graphs, heads, width], "
            f"got {vectors.shape}"
        )
    if labels.shape != vectors.shape[:2]:
        raise ValueError(
            f"layer PCA labels have shape {labels.shape}, expected {vectors.shape[:2]}"
        )
    num_graphs, num_heads, _ = vectors.shape
    default_rows, default_columns = _layer_pca_grid_shape(num_heads)
    nrows = default_rows if nrows is None else int(nrows)
    ncols = default_columns if ncols is None else int(ncols)
    if nrows < 1 or ncols < 1 or nrows * ncols < num_heads:
        raise ValueError(
            f"{nrows}x{ncols} grid cannot contain {num_heads} heads"
        )
    n_used = int(payload.get("n_used", num_graphs))
    if n_used != num_graphs:
        raise ValueError(
            f"layer PCA n_used={n_used} but vectors contain {num_graphs} graphs"
        )

    categories = _ordered_pca_categories(
        [str(label) for label in labels.reshape(-1)]
    )
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(max(9.0, 2.25 * ncols), 2.15 * nrows + 1.55),
        squeeze=False,
    )
    flat_axes = axes.reshape(-1)
    for head, ax in enumerate(flat_axes[:num_heads]):
        coordinates, explained = _pca(vectors[:, head, :])
        head_labels = [str(label) for label in labels[:, head]]
        colors = [_pca_focus_color(label) for label in head_labels]
        ax.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            s=9,
            c=colors,
            edgecolors="white",
            linewidths=0.18,
            alpha=0.76,
        )
        ax.axhline(0, color=LIGHT_GRID, linewidth=0.6, zorder=0)
        ax.axvline(0, color=LIGHT_GRID, linewidth=0.6, zorder=0)
        ax.set_title(
            f"H{head}\n"
            f"PC1 {100 * explained[0]:.0f}% · PC2 {100 * explained[1]:.0f}%",
            fontsize=10,
            pad=3,
        )
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.tick_params(labelsize=8, length=2.5, pad=1.5)
        row, column = divmod(head, ncols)
        if row == nrows - 1:
            ax.set_xlabel("PC1", fontsize=9)
        else:
            ax.tick_params(labelbottom=False)
        if column == 0:
            ax.set_ylabel("PC2", fontsize=9)
        else:
            ax.tick_params(labelleft=False)
        ax.grid(False)
    for ax in flat_axes[num_heads:]:
        ax.axis("off")

    handles = [
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker="o",
            markersize=7,
            markerfacecolor=_pca_focus_color(category),
            markeredgecolor="white",
            markeredgewidth=0.4,
            label=category,
        )
        for category in categories
    ]
    legend_columns = min(7, max(1, len(handles)))
    legend = fig.legend(
        handles=handles,
        labels=categories,
        title="Attention focus",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.018),
        ncol=legend_columns,
        frameon=False,
        fontsize=10.5,
        title_fontsize=12,
        handletextpad=0.5,
        columnspacing=1.25,
        borderaxespad=0,
    )
    legend.get_title().set_color(NAVY)
    fig.suptitle(
        f"{_payload_display_title(payload)}\n"
        f"PCA of routed head output — Layer {layer} (all heads); "
        f"$n = {n_used}$ molecules",
        fontsize=18,
        y=0.985,
    )
    fig.subplots_adjust(
        left=0.055,
        right=0.992,
        top=0.84,
        bottom=0.20,
        wspace=0.24,
        hspace=0.36,
    )
    return fig


def plot_hop_attention_mass(
    metrics: CanonicalHeadMetrics,
    head: Head,
    *,
    title: str | None = None,
):
    if metrics.clean_attention_distance is None:
        raise ValueError("score cache has no clean_attention_distance profile")
    apply_publication_style()
    layer, index = head
    values = np.asarray(metrics.clean_attention_distance[layer, index])
    labels = list(metrics.distance_axis)
    special = np.asarray(
        [
            str(label).lower().replace(" ", "_") in {"graph_token", "virtual"}
            for label in labels
        ]
    )
    x = np.arange(len(values), dtype=np.float64)
    x[special] += 0.8
    colors = [GOLD if value else TEAL for value in special]
    fig, ax = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    bars = ax.bar(
        x,
        values,
        width=0.72,
        color=colors,
        edgecolor="white",
        linewidth=0.7,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [str(label).replace("_", "\n") for label in labels]
    )
    ax.set_xlabel("Shortest-path distance")
    ax.set_ylabel("Mean clean attention mass")
    ax.set_ylim(0, max(float(values.max()) * 1.18, 0.05))
    ax.set_title(
        title or f"Attention mass by hop distance — {_head_label(head)}",
        fontsize=15,
    )
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        if value >= 0.025:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=NAVY,
            )
    return fig


def plot_logit_spread(
    logit_payload: Mapping[str, Any],
    *,
    title: str = "GRIT raw attention-logit spread",
):
    """GRIT node-only counterfactual and actual relation-conditioned spread."""

    apply_publication_style()
    node = np.asarray(logit_payload["node_std_mean"])
    node_error = np.asarray(logit_payload["node_std_std"])
    relation = np.asarray(logit_payload["relation_std_mean"])
    relation_error = np.asarray(logit_payload["relation_std_std"])
    layers = np.arange(len(node))
    fig, ax = plt.subplots(figsize=(8.2, 4.9), constrained_layout=True)
    ax.fill_between(
        layers,
        np.clip(node - node_error, 0.0, None),
        node + node_error,
        color=BLUE,
        alpha=0.18,
        linewidth=0,
    )
    ax.plot(
        layers,
        node,
        color=BLUE,
        marker="o",
        linewidth=2.0,
        markersize=5,
        label="Node-only counterfactual: mean ±1 SD",
    )
    ax.fill_between(
        layers,
        np.clip(relation - relation_error, 0.0, None),
        relation + relation_error,
        color=ORANGE,
        alpha=0.16,
        linewidth=0,
    )
    ax.plot(
        layers,
        relation,
        color=ORANGE,
        marker="s",
        linewidth=2.0,
        markersize=5,
        label="Relation-conditioned actual: mean ±1 SD",
    )
    ax.set_xticks(layers)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Mean key-wise raw-logit standard deviation")
    ax.set_title(
        f"{_payload_display_title(logit_payload)}\n"
        f"{title}\n$n={int(logit_payload['n_used'])}$ molecules",
        fontsize=15,
    )
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend()
    return fig


def _average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    return ranks


def _spearman_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_rank, right_rank = _average_ranks(left), _average_ranks(right)
    if np.std(left_rank) == 0 or np.std(right_rank) == 0:
        return float("nan")
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def plot_selectivity_vs_logit_ratio(
    metrics: CanonicalHeadMetrics,
    logit_payload: Mapping[str, Any],
    selected_heads: Mapping[str, Head] | None = None,
    *,
    active_only: bool = True,
):
    """``D_rel`` versus GRIT node-only/relation-conditioned logit balance."""

    apply_publication_style()
    ratio = np.asarray(logit_payload["log_r_mean"], dtype=np.float64)
    if ratio.shape != metrics.shape:
        raise ValueError(
            f"logit ratio shape {ratio.shape} != head metric grid {metrics.shape}"
        )
    selectivity = metrics.selectivity
    finite = np.isfinite(ratio) & np.isfinite(selectivity)
    if active_only:
        finite &= metrics.active
    correlation = (
        _spearman_correlation(ratio[finite], selectivity[finite])
        if finite.sum() >= 3
        else float("nan")
    )
    correlation_label = (
        rf"; active-head Spearman $\rho={correlation:.2f}$"
        if np.isfinite(correlation)
        else ""
    )
    fig, ax = plt.subplots(figsize=(8.1, 5.8), constrained_layout=True)
    scatter = _scatter_heads(
        ax,
        ratio,
        selectivity,
        active=metrics.active if active_only else None,
    )
    ax.axhline(0, color=SLATE, linestyle="--", linewidth=1.0)
    ax.axvline(0, color=SLATE, linestyle="--", linewidth=1.0)
    ax.set_xlabel(
        r"$\log_{10}[\mathrm{std}(\mathrm{node\ only})/"
        r"\mathrm{std}(\mathrm{relation\ conditioned})]$"
    )
    ax.set_ylabel(r"Relative selectivity $D_{\rm rel}$")
    ax.set_title(
        f"{_payload_display_title(logit_payload)}\n"
        "Relative selectivity versus GRIT logit balance\n"
        f"$n={int(logit_payload['n_used'])}$ molecules"
        f"{correlation_label}",
        fontsize=15,
    )
    ax.grid(True)
    _annotate_selected_scatter(ax, ratio, selectivity, selected_heads)
    if active_only and np.any(~metrics.active):
        ax.legend(loc="lower left", fontsize=9)
    colorbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    colorbar.set_label("Layer")
    colorbar.set_ticks(np.arange(metrics.num_layers))
    return fig


def save_figure_bundle(
    figure,
    output_directory: str | Path,
    stem: str,
    *,
    metadata: Mapping[str, Any] | None = None,
    dpi: int = PUBLICATION_PNG_DPI,
    pdf_dpi: int = PUBLICATION_PDF_RASTER_DPI,
    supersede_stem_globs: Sequence[str] = (),
) -> dict[str, Path]:
    """Save a publication PNG, hybrid-vector PDF, and provenance sidecar.

    ``supersede_stem_globs`` removes replaceable figure bundles only after the
    new bundle has been written successfully. Patterns are restricted to direct
    children of the output directory and to the three generated file types.
    """

    dpi = int(dpi)
    pdf_dpi = int(pdf_dpi)
    if dpi < 1 or pdf_dpi < 1:
        raise ValueError("PNG and PDF raster DPI must both be positive")
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    supersede_patterns = tuple(str(pattern) for pattern in supersede_stem_globs)
    for pattern in supersede_patterns:
        if Path(pattern).name != pattern:
            raise ValueError(
                "superseded figure patterns must be direct filename globs, "
                f"got {pattern!r}"
            )
    paths = {
        "png": directory / f"{stem}.png",
        "pdf": directory / f"{stem}.pdf",
        "metadata": directory / f"{stem}.json",
    }
    figure.savefig(
        paths["png"],
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.04,
        facecolor="white",
    )
    figure.savefig(
        paths["pdf"],
        dpi=pdf_dpi,
        bbox_inches="tight",
        pad_inches=0.04,
        facecolor="white",
        metadata={
            "Title": str(stem),
            "Creator": "Graph Specialisation and Metrics",
            "Subject": "Publication figure",
        },
    )
    export_metadata = {
        "png_dpi": dpi,
        "pdf_raster_dpi": pdf_dpi,
        "pdf_vector_text_and_paths": True,
        "pdf_font_embedding": "TrueType (fonttype 42)",
        "molecule_render_dpi": MOLECULE_RENDER_DPI,
    }
    paths["metadata"].write_text(
        json.dumps(
            {
                "figure": stem,
                "export_quality": export_metadata,
                **dict(metadata or {}),
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    current_paths = {path.resolve() for path in paths.values()}
    generated_suffixes = {".png", ".pdf", ".json"}
    for pattern in supersede_patterns:
        for candidate in directory.glob(pattern):
            if (
                candidate.is_file()
                and candidate.suffix.lower() in generated_suffixes
                and candidate.resolve() not in current_paths
            ):
                candidate.unlink()
    return paths


def save_section_pdf_bundles(
    section_pages: Mapping[str, Sequence[tuple[str, str | Path]]],
    output_directory: str | Path,
    *,
    task_prefix: str,
    section_titles: Mapping[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Merge ordered individual figure PDFs into task-prefixed section PDFs."""

    from pypdf import PdfReader, PdfWriter

    prefix = str(task_prefix).strip().lower()
    if not prefix or not prefix.replace("_", "").replace("-", "").isalnum():
        raise ValueError(f"invalid PDF task prefix {task_prefix!r}")
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, dict[str, Any]] = {}
    for section, entries in section_pages.items():
        section = str(section)
        if not entries:
            continue
        if not section.replace("_", "").replace("-", "").isalnum():
            raise ValueError(f"invalid PDF section name {section!r}")
        writer = PdfWriter()
        stems: list[str] = []
        page_count = 0
        try:
            for stem, source in entries:
                source_path = Path(source)
                if not source_path.is_file():
                    raise FileNotFoundError(
                        f"section PDF source does not exist: {source_path}"
                    )
                reader = PdfReader(str(source_path))
                for page in reader.pages:
                    writer.add_page(page)
                    page_count += 1
                stems.append(str(stem))
            if not page_count:
                raise ValueError(f"PDF section {section!r} has no pages")
            title = str((section_titles or {}).get(section, section))
            writer.add_metadata(
                {
                    "/Title": f"{prefix.upper()} — {title}",
                    "/Subject": "Graph specialisation figure section",
                }
            )
            output_path = directory / f"{prefix}_{section}.pdf"
            descriptor, temporary = tempfile.mkstemp(
                prefix=output_path.stem + "-",
                suffix=".partial.pdf",
                dir=directory,
            )
            os.close(descriptor)
            temporary_path = Path(temporary)
            try:
                with temporary_path.open("wb") as handle:
                    writer.write(handle)
                temporary_path.replace(output_path)
            finally:
                if temporary_path.exists():
                    temporary_path.unlink()
        finally:
            writer.close()
        outputs[section] = {
            "path": output_path,
            "pages": page_count,
            "figure_stems": stems,
        }
    return outputs


__all__ = [
    "MOLECULE_RENDER_DPI",
    "PUBLICATION_PDF_RASTER_DPI",
    "PUBLICATION_PNG_DPI",
    "apply_publication_style",
    "automatic_selectivity_limits",
    "plot_attention_grid",
    "plot_av_pca",
    "plot_coordinate_heatmaps",
    "plot_hop_attention_mass",
    "plot_layer_av_pca_grid",
    "plot_logit_spread",
    "plot_score_heatmaps",
    "plot_score_plane",
    "plot_selectivity_joint_plane",
    "plot_selectivity_vs_logit_ratio",
    "save_figure_bundle",
    "save_section_pdf_bundles",
]
