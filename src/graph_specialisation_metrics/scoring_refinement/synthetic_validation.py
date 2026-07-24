"""Pure multi-method validation and plotting for the mixed synthetic task.

This module deliberately knows nothing about the GRIT checkpoint or the synthetic
data generator.  A task adapter supplies per-seed score matrices and cached causal
measurements; the functions below turn those inputs into one stable comparison
table and the four predeclared figure families.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


EPS = 1.0e-12
SYNTHETIC_METHODS = (
    "M1_DD",
    "M1_DT",
    "M1_TD",
    "M1_TT",
    "M4",
    "M5",
    "M7",
)
METHOD_LABELS = {
    "M1_DD": "M1_DD\nsemantic donor · PE donor",
    "M1_DT": "M1_DT\nsemantic donor · PE transpose",
    "M1_TD": "M1_TD\nsemantic transpose · PE donor",
    "M1_TT": "M1_TT\nsemantic transpose · PE transpose",
    "M4": "M4\nsemantic attention follow · invariant",
    "M5": "M5\nPE attention invariant · follow",
    "M7": "M7\nsemantic follow · PE follow",
}


def method_score_axes(
    components: Mapping[str, np.ndarray],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Map shared raw components to semantic/structural axes for all methods."""

    axes = {
        "M1_DD": (
            components["eg_semantic_single"],
            components["eg_pe_single"],
        ),
        "M1_DT": (
            components["eg_semantic_single"],
            components["eg_pe_transposition"],
        ),
        "M1_TD": (
            components["eg_semantic_transposition"],
            components["eg_pe_single"],
        ),
        "M1_TT": (
            components["eg_semantic_transposition"],
            components["eg_pe_transposition"],
        ),
        "M4": (
            components["semantic_attention_follow"],
            components["semantic_attention_invariant"],
        ),
        "M5": (
            components["pe_attention_invariant"],
            components["pe_attention_follow"],
        ),
        "M7": (
            components["semantic_attention_follow"],
            components["pe_attention_follow"],
        ),
    }
    shape = None
    for method, pair in axes.items():
        for value in pair:
            value = np.asarray(value, dtype=float)
            if value.ndim != 2:
                raise ValueError(f"{method} score axes must be [layer, head]")
            shape = value.shape if shape is None else shape
            if value.shape != shape:
                raise ValueError("all synthetic method axes must share one shape")
            if not np.isfinite(value).all():
                raise ValueError(f"{method} contains non-finite scores")
    return axes


def calibrate_axes(
    semantic: Any,
    structural: Any,
    *,
    semantic_reference: float | None = None,
    structural_reference: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mean-calibrate channels within one checkpoint, then compute ``J``/``D_rel``."""

    semantic = np.asarray(semantic, dtype=float)
    structural = np.asarray(structural, dtype=float)
    semantic_reference = (
        float(np.mean(semantic))
        if semantic_reference is None
        else float(semantic_reference)
    )
    structural_reference = (
        float(np.mean(structural))
        if structural_reference is None
        else float(structural_reference)
    )
    semantic_norm = semantic / max(semantic_reference, EPS)
    structural_norm = structural / max(structural_reference, EPS)
    total = semantic_norm + structural_norm
    joint = 0.5 * total
    selectivity = (semantic_norm - structural_norm) / (total + EPS)
    return semantic_norm, structural_norm, joint, selectivity


def select_head_groups(
    semantic: Any,
    structural: Any,
    *,
    size: int,
    semantic_reference: float | None = None,
    structural_reference: float | None = None,
) -> dict[str, list[tuple[int, int]]]:
    """Select disjoint active semantic/structural families from score only."""

    semantic_norm, structural_norm, joint, selectivity = calibrate_axes(
        semantic,
        structural,
        semantic_reference=semantic_reference,
        structural_reference=structural_reference,
    )
    eligible = joint >= np.quantile(joint, 0.40)
    heads = [
        (layer, head)
        for layer in range(int(joint.shape[0]))
        for head in range(int(joint.shape[1]))
    ]
    semantic_order = sorted(
        heads, key=lambda item: float(selectivity[item]), reverse=True
    )
    structural_order = sorted(
        heads, key=lambda item: float(selectivity[item])
    )
    semantic_group = [
        item for item in semantic_order if bool(eligible[item])
    ][: int(size)]
    structural_group = [
        item
        for item in structural_order
        if bool(eligible[item]) and item not in semantic_group
    ][: int(size)]
    if len(structural_group) < int(size):
        structural_group.extend(
            item
            for item in structural_order
            if item not in semantic_group and item not in structural_group
        )
        structural_group = structural_group[: int(size)]
    return {
        "semantic": semantic_group,
        "structural": structural_group,
    }


def rank_values(values: Any) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    for index in np.flatnonzero(counts > 1):
        positions = np.flatnonzero(inverse == index)
        ranks[positions] = float(np.mean(ranks[positions]))
    return ranks


def spearman(x: Iterable[float], y: Iterable[float]) -> float:
    x = np.asarray(list(x), dtype=float)
    y = np.asarray(list(y), dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < 3:
        return float("nan")
    x_rank = rank_values(x[valid])
    y_rank = rank_values(y[valid])
    if float(np.std(x_rank)) <= 0.0 or float(np.std(y_rank)) <= 0.0:
        return float("nan")
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def _within_layer_spearman(
    rows: Sequence[Mapping[str, Any]],
    x_key: str,
    y_key: str,
) -> float:
    """Rank within each seed/layer, center, then correlate the pooled residuals."""

    x_residuals: list[float] = []
    y_residuals: list[float] = []
    groups = sorted({(int(row["seed"]), int(row["layer"])) for row in rows})
    for seed, layer in groups:
        selected = [
            row
            for row in rows
            if int(row["seed"]) == seed and int(row["layer"]) == layer
        ]
        x = np.asarray([float(row[x_key]) for row in selected], dtype=float)
        y = np.asarray([float(row[y_key]) for row in selected], dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        if int(valid.sum()) < 2:
            continue
        x_rank = rank_values(x[valid])
        y_rank = rank_values(y[valid])
        x_residuals.extend((x_rank - x_rank.mean()).tolist())
        y_residuals.extend((y_rank - y_rank.mean()).tolist())
    if len(x_residuals) < 3:
        return float("nan")
    x_array = np.asarray(x_residuals)
    y_array = np.asarray(y_residuals)
    if float(np.std(x_array)) <= 0.0 or float(np.std(y_array)) <= 0.0:
        return float("nan")
    return float(np.corrcoef(x_array, y_array)[0, 1])


def correlation_summary(
    rows: Sequence[Mapping[str, Any]],
    x_key: str,
    y_key: str,
) -> dict[str, float]:
    pooled = spearman(
        [float(row[x_key]) for row in rows],
        [float(row[y_key]) for row in rows],
    )
    seeds = sorted({int(row["seed"]) for row in rows})
    per_seed = []
    for seed in seeds:
        selected = [row for row in rows if int(row["seed"]) == seed]
        per_seed.append(
            spearman(
                [float(row[x_key]) for row in selected],
                [float(row[y_key]) for row in selected],
            )
        )
    finite = np.asarray(per_seed, dtype=float)
    finite = finite[np.isfinite(finite)]
    return {
        "pooled_spearman": pooled,
        "within_layer_spearman": _within_layer_spearman(rows, x_key, y_key),
        "seed_median_spearman": (
            float(np.median(finite)) if finite.size else float("nan")
        ),
    }


def _finite_mean(value: Any) -> float:
    array = np.asarray(value, dtype=float)
    valid = np.isfinite(array)
    return float(array[valid].mean()) if bool(valid.any()) else float("nan")


def _group_mean(
    matrix: Any,
    heads: Sequence[Sequence[int] | tuple[int, int]],
) -> float:
    values = [
        np.asarray(matrix[int(layer), int(head)], dtype=float).reshape(-1)
        for layer, head in heads
    ]
    return _finite_mean(np.concatenate(values)) if values else float("nan")


def build_per_head_rows(
    results: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Create one row per seed/method/head using shared causal measurements."""

    rows: list[dict[str, Any]] = []
    for result in results:
        seed = int(result["seed"])
        m1_semantic_reference = float(
            np.mean(result["method_scores"]["M1_DT"]["semantic"])
        )
        m1_structural_reference = float(
            np.mean(result["method_scores"]["M1_DT"]["structural"])
        )
        semantic_ablation = np.asarray(
            result["ablation_semantic"]["functional"], dtype=float
        ).mean(axis=-1)
        structural_ablation = np.asarray(
            result["ablation_structural"]["functional"], dtype=float
        ).mean(axis=-1)
        semantic_ablation_norm = semantic_ablation / max(
            float(semantic_ablation.mean()), EPS
        )
        structural_ablation_norm = structural_ablation / max(
            float(structural_ablation.mean()), EPS
        )
        ablation_total = semantic_ablation_norm + structural_ablation_norm
        ablation_joint = 0.5 * ablation_total
        ablation_role = (
            semantic_ablation_norm - structural_ablation_norm
        ) / (ablation_total + EPS)
        semantic_rescue = np.nanmean(
            np.asarray(result["rescue_semantic"]["mediation"], dtype=float),
            axis=-1,
        )
        structural_rescue = np.nanmean(
            np.asarray(result["rescue_structural"]["mediation"], dtype=float),
            axis=-1,
        )
        rescue_role = semantic_rescue - structural_rescue
        for method in SYNTHETIC_METHODS:
            score = result["method_scores"][method]
            semantic = np.asarray(score["semantic"], dtype=float)
            structural = np.asarray(score["structural"], dtype=float)
            shared_m1_reference = method.startswith("M1_")
            sem_norm, str_norm, joint, selectivity = calibrate_axes(
                semantic,
                structural,
                semantic_reference=(
                    m1_semantic_reference if shared_m1_reference else None
                ),
                structural_reference=(
                    m1_structural_reference if shared_m1_reference else None
                ),
            )
            for layer in range(int(semantic.shape[0])):
                for head in range(int(semantic.shape[1])):
                    rows.append(
                        {
                            "seed": seed,
                            "method": method,
                            "layer": layer,
                            "head": head,
                            "semantic_score": float(semantic[layer, head]),
                            "structural_score": float(structural[layer, head]),
                            "semantic_reference": (
                                m1_semantic_reference
                                if shared_m1_reference
                                else float(np.mean(semantic))
                            ),
                            "structural_reference": (
                                m1_structural_reference
                                if shared_m1_reference
                                else float(np.mean(structural))
                            ),
                            "semantic_score_norm": float(sem_norm[layer, head]),
                            "structural_score_norm": float(str_norm[layer, head]),
                            "J": float(joint[layer, head]),
                            "D_rel": float(selectivity[layer, head]),
                            "semantic_ablation_movement": float(
                                semantic_ablation[layer, head]
                            ),
                            "structural_ablation_movement": float(
                                structural_ablation[layer, head]
                            ),
                            "necessity_joint": float(ablation_joint[layer, head]),
                            "necessity_role": float(ablation_role[layer, head]),
                            "semantic_rescue": float(semantic_rescue[layer, head]),
                            "structural_rescue": float(
                                structural_rescue[layer, head]
                            ),
                            "rescue_role": float(rescue_role[layer, head]),
                        }
                    )
    return rows


def family_validation(
    results: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, np.ndarray]]]:
    """Return seed-level 2x2 necessity/rescue matrices and across-seed means."""

    records: list[dict[str, Any]] = []
    aggregate: dict[str, dict[str, list[np.ndarray]]] = {
        method: {"necessity": [], "rescue": []} for method in SYNTHETIC_METHODS
    }
    for result in results:
        for method in SYNTHETIC_METHODS:
            groups = result["method_selected_groups"][method]
            family = result["method_family_ablation"][method]["tasks"]
            necessity = np.zeros((2, 2), dtype=float)
            rescue = np.zeros((2, 2), dtype=float)
            for row_index, family_name in enumerate(("semantic", "structural")):
                heads = groups[family_name]
                for column, task_name in enumerate(("semantic", "structural")):
                    curve = np.asarray(
                        family[task_name]["families"][family_name]["loss"],
                        dtype=float,
                    )
                    necessity[row_index, column] = _finite_mean(curve[-1])
                rescue[row_index, 0] = _group_mean(
                    result["rescue_semantic"]["mediation"], heads
                )
                rescue[row_index, 1] = _group_mean(
                    result["rescue_structural"]["mediation"], heads
                )
            necessity_interaction = (
                necessity[0, 0]
                - necessity[0, 1]
                + necessity[1, 1]
                - necessity[1, 0]
            )
            rescue_interaction = (
                rescue[0, 0]
                - rescue[0, 1]
                + rescue[1, 1]
                - rescue[1, 0]
            )
            records.append(
                {
                    "seed": int(result["seed"]),
                    "method": method,
                    "necessity_semantic_family_semantic_task": necessity[0, 0],
                    "necessity_semantic_family_structural_task": necessity[0, 1],
                    "necessity_structural_family_semantic_task": necessity[1, 0],
                    "necessity_structural_family_structural_task": necessity[1, 1],
                    "necessity_interaction": necessity_interaction,
                    "rescue_semantic_family_semantic_task": rescue[0, 0],
                    "rescue_semantic_family_structural_task": rescue[0, 1],
                    "rescue_structural_family_semantic_task": rescue[1, 0],
                    "rescue_structural_family_structural_task": rescue[1, 1],
                    "rescue_interaction": rescue_interaction,
                }
            )
            aggregate[method]["necessity"].append(necessity)
            aggregate[method]["rescue"].append(rescue)
    means = {
        method: {
            key: np.nanmean(np.stack(value, axis=0), axis=0)
            for key, value in families.items()
        }
        for method, families in aggregate.items()
    }
    return records, means


def _configure_matplotlib() -> Any:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 130,
            "savefig.dpi": 320,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return plt


def _save_figure(fig: Any, base: Path) -> list[str]:
    base.parent.mkdir(parents=True, exist_ok=True)
    paths = [base.with_suffix(".png"), base.with_suffix(".pdf")]
    for path in paths:
        fig.savefig(path)
    _configure_matplotlib().close(fig)
    return [str(path) for path in paths]


def _palette(layers: int) -> list[Any]:
    plt = _configure_matplotlib()
    cmap = plt.get_cmap("viridis")
    return [cmap(index / max(layers - 1, 1)) for index in range(layers)]


def _scatter(
    ax: Any,
    rows: Sequence[Mapping[str, Any]],
    x_key: str,
    y_key: str,
    *,
    layers: int,
) -> None:
    colors = _palette(layers)
    markers = ("o", "s", "^", "D", "P")
    seeds = sorted({int(row["seed"]) for row in rows})
    for row in rows:
        seed_index = seeds.index(int(row["seed"]))
        ax.scatter(
            float(row[x_key]),
            float(row[y_key]),
            color=colors[int(row["layer"])],
            marker=markers[seed_index % len(markers)],
            s=28,
            alpha=0.82,
            edgecolor="white",
            linewidth=0.35,
        )
    ax.grid(True, linewidth=0.45, alpha=0.20)


def _legend(fig: Any, rows: Sequence[Mapping[str, Any]], layers: int) -> None:
    from matplotlib.lines import Line2D

    colors = _palette(layers)
    markers = ("o", "s", "^", "D", "P")
    seeds = sorted({int(row["seed"]) for row in rows})
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=colors[layer],
            markeredgecolor="none",
            label=f"Layer {layer}",
        )
        for layer in range(layers)
    ]
    handles.extend(
        Line2D(
            [0],
            [0],
            marker=markers[index % len(markers)],
            linestyle="none",
            markerfacecolor="none",
            markeredgecolor="#555555",
            label=f"Seed {seed}",
        )
        for index, seed in enumerate(seeds)
    )
    fig.legend(
        handles=handles,
        frameon=False,
        loc="lower center",
        ncol=min(len(handles), 8),
        bbox_to_anchor=(0.5, 0.005),
    )


def figure_score_planes(
    rows: Sequence[Mapping[str, Any]],
    *,
    layers: int,
    output_dir: Path,
) -> list[str]:
    plt = _configure_matplotlib()
    fig, axes = plt.subplots(2, 4, figsize=(15.0, 7.4))
    for ax, method in zip(axes.ravel(), SYNTHETIC_METHODS):
        selected = [row for row in rows if row["method"] == method]
        _scatter(
            ax,
            selected,
            "structural_score_norm",
            "semantic_score_norm",
            layers=layers,
        )
        limits = np.asarray(
            [
                float(row[key])
                for row in selected
                for key in ("structural_score_norm", "semantic_score_norm")
            ]
        )
        lower = max(0.0, float(np.nanmin(limits)) * 0.92)
        upper = float(np.nanmax(limits)) * 1.06
        ax.plot([lower, upper], [lower, upper], "--", color="#999999", linewidth=0.8)
        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(METHOD_LABELS[method])
        ax.set_xlabel("Structural score / seed mean")
        ax.set_ylabel("Semantic score / seed mean")
    axes.ravel()[-1].axis("off")
    _legend(fig, rows, layers)
    fig.suptitle(
        "Semantic and structural head-score planes across candidate methods",
        x=0.035,
        y=0.995,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.075, 1, 0.94), h_pad=2.0, w_pad=1.4)
    return _save_figure(fig, output_dir / "fig1_all_methods_score_planes")


def figure_necessity_and_role(
    rows: Sequence[Mapping[str, Any]],
    *,
    layers: int,
    output_dir: Path,
) -> tuple[list[str], list[dict[str, Any]]]:
    plt = _configure_matplotlib()
    fig, axes = plt.subplots(2, len(SYNTHETIC_METHODS), figsize=(20.0, 7.2))
    summaries: list[dict[str, Any]] = []
    for column, method in enumerate(SYNTHETIC_METHODS):
        selected = [row for row in rows if row["method"] == method]
        joint = correlation_summary(selected, "J", "necessity_joint")
        role = correlation_summary(selected, "D_rel", "necessity_role")
        summaries.extend(
            [
                {"method": method, "validation": "J_vs_necessity", **joint},
                {"method": method, "validation": "D_vs_necessity_role", **role},
            ]
        )
        ax = axes[0, column]
        _scatter(ax, selected, "J", "necessity_joint", layers=layers)
        ax.set_title(
            f"{method}\nρ={joint['pooled_spearman']:.2f}; "
            f"within={joint['within_layer_spearman']:.2f}"
        )
        ax.set_xlabel("Score strength J")
        if column == 0:
            ax.set_ylabel("Joint necessity\n(mean calibrated ablation movement)")
        ax = axes[1, column]
        _scatter(ax, selected, "D_rel", "necessity_role", layers=layers)
        ax.axhline(0.0, color="#888888", linewidth=0.7)
        ax.axvline(0.0, color="#888888", linewidth=0.7)
        ax.set_xlim(-1.04, 1.04)
        ax.set_ylim(-1.04, 1.04)
        ax.set_title(
            f"ρ={role['pooled_spearman']:.2f}; "
            f"within={role['within_layer_spearman']:.2f}"
        )
        ax.set_xlabel(r"Score role $D_{rel}$")
        if column == 0:
            ax.set_ylabel("Necessity role\n(semantic − structural)")
    _legend(fig, rows, layers)
    fig.suptitle(
        "Does each method predict head necessity and which task the head serves?",
        x=0.025,
        y=0.995,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.025,
        0.955,
        "Top: J versus held-out clean-input ablation movement. "
        "Bottom: D_rel versus the semantic–structural ablation-role contrast.",
        color="#555555",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.085, 1, 0.92), h_pad=2.1, w_pad=1.3)
    paths = _save_figure(
        fig, output_dir / "fig2_all_methods_necessity_and_role"
    )
    return paths, summaries


def figure_rescue_role(
    rows: Sequence[Mapping[str, Any]],
    *,
    layers: int,
    output_dir: Path,
) -> tuple[list[str], list[dict[str, Any]]]:
    plt = _configure_matplotlib()
    fig, axes = plt.subplots(1, len(SYNTHETIC_METHODS), figsize=(20.0, 3.7))
    summaries = []
    for ax, method in zip(axes, SYNTHETIC_METHODS):
        selected = [row for row in rows if row["method"] == method]
        summary = correlation_summary(selected, "D_rel", "rescue_role")
        summaries.append(
            {"method": method, "validation": "D_vs_rescue_role", **summary}
        )
        _scatter(ax, selected, "D_rel", "rescue_role", layers=layers)
        ax.axhline(0.0, color="#888888", linewidth=0.7)
        ax.axvline(0.0, color="#888888", linewidth=0.7)
        ax.set_xlim(-1.04, 1.04)
        ax.set_title(
            f"{method}\nρ={summary['pooled_spearman']:.2f}; "
            f"within={summary['within_layer_spearman']:.2f}"
        )
        ax.set_xlabel(r"Score role $D_{rel}$")
    axes[0].set_ylabel("Causal rescue role\n(semantic − structural mediation)")
    _legend(fig, rows, layers)
    fig.suptitle(
        "Does score selectivity predict the task rescued by a head?",
        x=0.025,
        y=0.995,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.16, 1, 0.88), w_pad=1.4)
    paths = _save_figure(fig, output_dir / "fig3_all_methods_rescue_role")
    return paths, summaries


def _draw_matrix(ax: Any, matrix: np.ndarray, *, title: str) -> None:
    import matplotlib.colors as colors

    maximum = max(float(np.nanmax(np.abs(matrix))), 1.0e-6)
    image = ax.imshow(
        matrix,
        cmap="RdBu_r",
        norm=colors.TwoSlopeNorm(vmin=-maximum, vcenter=0.0, vmax=maximum),
        aspect="equal",
    )
    ax.set_xticks((0, 1), ("Semantic", "Structural"), rotation=24, ha="right")
    ax.set_yticks((0, 1), ("Sem. heads", "Str. heads"))
    ax.set_title(title)
    for row in range(2):
        for column in range(2):
            value = float(matrix[row, column])
            ax.text(
                column,
                row,
                f"{value:+.3f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if abs(value) > 0.58 * maximum else "black",
                fontweight="bold",
            )
    ax.figure.colorbar(image, ax=ax, fraction=0.047, pad=0.03)


def figure_necessity_rescue(
    family_means: Mapping[str, Mapping[str, np.ndarray]],
    *,
    output_dir: Path,
) -> list[str]:
    plt = _configure_matplotlib()
    fig, axes = plt.subplots(2, len(SYNTHETIC_METHODS), figsize=(20.0, 7.0))
    for column, method in enumerate(SYNTHETIC_METHODS):
        necessity = np.asarray(family_means[method]["necessity"], dtype=float)
        rescue = np.asarray(family_means[method]["rescue"], dtype=float)
        necessity_interaction = (
            necessity[0, 0]
            - necessity[0, 1]
            + necessity[1, 1]
            - necessity[1, 0]
        )
        rescue_interaction = (
            rescue[0, 0] - rescue[0, 1] + rescue[1, 1] - rescue[1, 0]
        )
        _draw_matrix(
            axes[0, column],
            necessity,
            title=f"{method}\nnecessity ΔCE\ninteraction={necessity_interaction:+.3f}",
        )
        _draw_matrix(
            axes[1, column],
            rescue,
            title=f"rescue mediation\ninteraction={rescue_interaction:+.3f}",
        )
    axes[0, 0].set_ylabel("Selected family")
    axes[1, 0].set_ylabel("Selected family")
    fig.suptitle(
        "Score-selected head families: necessity plus causal rescue",
        x=0.025,
        y=0.995,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.025,
        0.955,
        "Columns in each 2×2 matrix are task/corruption channels; rows are "
        "disjoint semantic- and structural-score-selected families.",
        color="#555555",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.92), h_pad=2.0, w_pad=1.3)
    return _save_figure(fig, output_dir / "fig4_all_methods_necessity_rescue")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def create_validation_outputs(
    results: Sequence[Mapping[str, Any]],
    *,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Write exactly four consolidated figure families plus auditable tables."""

    if not results:
        raise ValueError("at least one synthetic result is required")
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    rows = build_per_head_rows(results)
    layers = int(
        max(int(row["layer"]) for row in rows) + 1
    )
    family_rows, family_means = family_validation(results)
    fig1 = figure_score_planes(rows, layers=layers, output_dir=figures_dir)
    fig2, necessity_summary = figure_necessity_and_role(
        rows, layers=layers, output_dir=figures_dir
    )
    fig3, rescue_summary = figure_rescue_role(
        rows, layers=layers, output_dir=figures_dir
    )
    fig4 = figure_necessity_rescue(family_means, output_dir=figures_dir)
    validation_rows = necessity_summary + rescue_summary
    _write_csv(tables_dir / "per_head_all_methods.csv", rows)
    _write_csv(tables_dir / "method_validation_correlations.csv", validation_rows)
    _write_csv(tables_dir / "family_necessity_rescue.csv", family_rows)
    summary = {
        "methods": list(SYNTHETIC_METHODS),
        "seeds": [int(result["seed"]) for result in results],
        "figures": {
            "score_planes": fig1,
            "necessity_and_role": fig2,
            "rescue_role": fig3,
            "necessity_rescue": fig4,
        },
        "validation": validation_rows,
        "tables": {
            "per_head": str(tables_dir / "per_head_all_methods.csv"),
            "correlations": str(
                tables_dir / "method_validation_correlations.csv"
            ),
            "family": str(tables_dir / "family_necessity_rescue.csv"),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    return summary
