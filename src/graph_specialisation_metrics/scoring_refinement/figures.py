"""Clear, task-local figure atlas for all scoring methods."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


METHOD_ORDER = (
    "M1_DD",
    "M1_DT",
    "M1_TD",
    "M1_TT",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6",
)
RAW_AXIS_LABELS = {
    "M1_DD": ("Semantic single-donor output-projected EG", "PE single-copy output-projected EG"),
    "M1_DT": (
        "Semantic single-donor output-projected EG",
        "PE transposition output-projected EG",
    ),
    "M1_TD": (
        "Semantic transposition output-projected EG",
        "PE single-copy output-projected EG",
    ),
    "M1_TT": (
        "Semantic transposition output-projected EG",
        "PE transposition output-projected EG",
    ),
    "M2": ("Semantic transport-follow", "Semantic transport-invariant"),
    "M3": ("PE transport-invariant", "PE transport-follow"),
    "M4": ("Semantic attention-follow", "Semantic attention-invariant"),
    "M5": ("PE attention-invariant", "PE attention-follow"),
    "M6": ("Semantic transport-follow", "PE transport-follow"),
}


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 220,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 9,
        }
    )
    return plt


def _save(fig: Any, stem: Path) -> list[str]:
    paths = []
    for suffix in (".png", ".pdf"):
        path = stem.with_suffix(suffix)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
        paths.append(str(path))
    return paths


def _method_arrays(
    rows: Sequence[Mapping[str, Any]],
    method: str,
    left: str,
    right: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = [row for row in rows if str(row["method"]) == method]
    selected.sort(key=lambda row: (int(row["layer"]), int(row["head"])))
    return (
        np.asarray([float(row[left]) for row in selected]),
        np.asarray([float(row[right]) for row in selected]),
        np.asarray([int(row["layer"]) for row in selected]),
    )


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    from scipy.stats import spearmanr

    left = np.asarray(left, dtype=float).reshape(-1)
    right = np.asarray(right, dtype=float).reshape(-1)
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 3:
        return float("nan")
    left = left[valid]
    right = right[valid]
    if np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(spearmanr(left, right).statistic)


def _within_layer_spearman(
    left: np.ndarray,
    right: np.ndarray,
    layers: np.ndarray,
) -> float:
    values = [
        _spearman(left[layers == layer], right[layers == layer])
        for layer in sorted(set(layers.tolist()))
    ]
    finite = [value for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def _scatter(
    ax: Any,
    left: np.ndarray,
    right: np.ndarray,
    layers: np.ndarray,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    diagonal: bool,
) -> Any | None:
    left = np.asarray(left, dtype=float).reshape(-1)
    right = np.asarray(right, dtype=float).reshape(-1)
    layers = np.asarray(layers).reshape(-1)
    if not (len(left) == len(right) == len(layers)):
        raise ValueError(
            f"scatter inputs must align, got {len(left)}, {len(right)}, {len(layers)}"
        )
    valid = np.isfinite(left) & np.isfinite(right) & np.isfinite(layers)
    left = left[valid]
    right = right[valid]
    layers = layers[valid]
    if not len(left):
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.text(
            0.5,
            0.5,
            "No paired finite scores",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="0.45",
        )
        return None
    scatter = ax.scatter(
        left, right, c=layers, cmap="viridis", s=28, alpha=0.85, edgecolor="none"
    )
    if diagonal and len(left):
        low = float(min(np.nanmin(left), np.nanmin(right)))
        high = float(max(np.nanmax(left), np.nanmax(right)))
        ax.plot([low, high], [low, high], color="0.55", linewidth=0.8, linestyle="--")
    ax.set_title(
        f"{title}\npooled ρ={_spearman(left, right):.2f}; "
        f"within-layer ρ={_within_layer_spearman(left, right, layers):.2f}"
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return scatter


def _available_methods(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    present = {str(row["method"]) for row in rows}
    return [method for method in METHOD_ORDER if method in present]


def _disable_unused(axes: Sequence[Any], used: int) -> None:
    for ax in list(axes)[used:]:
        ax.axis("off")


def _paired_method_values(
    rows: Sequence[Mapping[str, Any]],
    left_method: str,
    right_method: str,
    field: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    def indexed(method: str) -> dict[tuple[int, int], float]:
        return {
            (int(row["layer"]), int(row["head"])): float(row[field])
            for row in rows
            if str(row["method"]) == method
        }

    left = indexed(left_method)
    right = indexed(right_method)
    keys = sorted(set(left) & set(right))
    return (
        np.asarray([left[key] for key in keys], dtype=float),
        np.asarray([right[key] for key in keys], dtype=float),
        np.asarray([key[0] for key in keys], dtype=int),
    )


def raw_score_atlas(rows: Sequence[Mapping[str, Any]], out: Path) -> list[str]:
    plt = _pyplot()
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 12.0), constrained_layout=True)
    scatter = None
    methods = _available_methods(rows)
    for ax, method in zip(axes.flat, methods):
        semantic, pe, layer = _method_arrays(
            rows, method, "semantic_score", "pe_score"
        )
        current = _scatter(
            ax,
            semantic,
            pe,
            layer,
            title=method,
            xlabel=RAW_AXIS_LABELS[method][0],
            ylabel=RAW_AXIS_LABELS[method][1],
            diagonal=method in {"M2", "M3", "M4", "M5"},
        )
        if current is not None:
            scatter = current
    _disable_unused(axes.flat, len(methods))
    if scatter is not None:
        fig.colorbar(scatter, ax=axes, label="Layer", shrink=0.65)
    paths = _save(fig, out / "raw_score_atlas")
    plt.close(fig)
    return paths


def m1_factorial(rows: Sequence[Mapping[str, Any]], out: Path) -> list[str]:
    plt = _pyplot()
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 8.5), constrained_layout=True)
    scatter = None
    methods = [
        method for method in METHOD_ORDER[:4] if method in _available_methods(rows)
    ]
    for ax, method in zip(axes.flat, methods):
        method_rows = [row for row in rows if str(row["method"]) == method]
        semantic, pe, layer = _method_arrays(
            rows, method, "semantic_score", "pe_score"
        )
        current = _scatter(
            ax,
            semantic,
            pe,
            layer,
            title=method,
            xlabel=str(method_rows[0]["semantic_intervention"]),
            ylabel=str(method_rows[0]["pe_intervention"]),
            diagonal=False,
        )
        if current is not None:
            scatter = current
    _disable_unused(axes.flat, len(methods))
    if scatter is not None:
        fig.colorbar(scatter, ax=axes, label="Layer", shrink=0.75)
    paths = _save(fig, out / "m1_intervention_factorial")
    plt.close(fig)
    return paths


def donor_vs_transposition(rows: Sequence[Mapping[str, Any]], out: Path) -> list[str]:
    plt = _pyplot()
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.5), constrained_layout=True)
    sem_donor, sem_trans, layer = _paired_method_values(
        rows, "M1_DD", "M1_TD", "semantic_score"
    )
    pe_donor, pe_trans, pe_layer = _paired_method_values(
        rows, "M1_DD", "M1_DT", "pe_score"
    )
    left_scatter = _scatter(
        axes[0, 0],
        sem_donor,
        sem_trans,
        layer,
        title="Semantic intervention agreement",
        xlabel="Single-node donor EG",
        ylabel="Node-transposition EG",
        diagonal=True,
    )
    right_scatter = _scatter(
        axes[0, 1],
        pe_donor,
        pe_trans,
        pe_layer,
        title="PE intervention agreement",
        xlabel="Single-node donor/copy EG",
        ylabel="Node-transposition EG",
        diagonal=True,
    )
    for ax, donor, transposition, item_layer, channel in (
        (axes[1, 0], sem_donor, sem_trans, layer, "Semantic"),
        (axes[1, 1], pe_donor, pe_trans, pe_layer, "PE"),
    ):
        donor_log = np.log1p(np.maximum(donor, 0.0))
        transposition_log = np.log1p(np.maximum(transposition, 0.0))
        mean_log = 0.5 * (donor_log + transposition_log)
        difference = transposition_log - donor_log
        ax.scatter(
            mean_log,
            difference,
            c=item_layer,
            cmap="viridis",
            s=28,
            alpha=0.85,
            edgecolor="none",
        )
        ax.axhline(0.0, color="0.55", linewidth=0.8, linestyle="--")
        ax.set_title(f"{channel}: log-scale difference")
        ax.set_xlabel("Mean log1p score")
        ax.set_ylabel("log1p(transposition) − log1p(donor)")
        if not len(donor):
            ax.text(
                0.5,
                0.5,
                "Required M1 arms not selected",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="0.45",
            )
    color_source = left_scatter if left_scatter is not None else right_scatter
    if color_source is not None:
        fig.colorbar(color_source, ax=axes, label="Layer", shrink=0.75)
    paths = _save(fig, out / "donor_vs_transposition")
    plt.close(fig)
    return paths


def derived_dj_atlas(rows: Sequence[Mapping[str, Any]], out: Path) -> list[str]:
    plt = _pyplot()
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 12.0), constrained_layout=True)
    scatter = None
    methods = _available_methods(rows)
    for ax, method in zip(axes.flat, methods):
        d_rel, joint, layer = _method_arrays(rows, method, "D_rel", "J")
        current = _scatter(
            ax,
            d_rel,
            joint,
            layer,
            title=method,
            xlabel="D_rel",
            ylabel="J",
            diagonal=False,
        )
        if current is not None:
            scatter = current
        ax.axvline(0.0, color="0.65", linewidth=0.8)
    _disable_unused(axes.flat, len(methods))
    if scatter is not None:
        fig.colorbar(scatter, ax=axes, label="Layer", shrink=0.65)
    paths = _save(fig, out / "derived_DJ_atlas")
    plt.close(fig)
    return paths


def _head_target(
    rows: Sequence[Mapping[str, Any]],
    key: str,
) -> dict[tuple[int, int], float]:
    return {
        (int(row["layer"]), int(row["head"])): float(row[key])
        for row in rows
    }


def significance_validation(
    derived_rows: Sequence[Mapping[str, Any]],
    ablation_rows: Sequence[Mapping[str, Any]],
    out: Path,
) -> list[str]:
    plt = _pyplot()
    target = _head_target(ablation_rows, "prediction_movement")
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 12.0), constrained_layout=True)
    scatter = None
    methods = _available_methods(derived_rows)
    for ax, method in zip(axes.flat, methods):
        rows = [row for row in derived_rows if row["method"] == method]
        rows.sort(key=lambda row: (int(row["layer"]), int(row["head"])))
        joint = np.asarray([float(row["J"]) for row in rows])
        effect = np.asarray(
            [
                target.get((int(row["layer"]), int(row["head"])), np.nan)
                for row in rows
            ]
        )
        layer = np.asarray([int(row["layer"]) for row in rows])
        current = _scatter(
            ax,
            joint,
            effect,
            layer,
            title=method,
            xlabel="Score strength J",
            ylabel="Held-out ablation movement",
            diagonal=False,
        )
        if current is not None:
            scatter = current
    _disable_unused(axes.flat, len(methods))
    if scatter is not None:
        fig.colorbar(scatter, ax=axes, label="Layer", shrink=0.65)
    paths = _save(fig, out / "significance_validation")
    plt.close(fig)
    return paths


def role_validation(
    derived_rows: Sequence[Mapping[str, Any]],
    causal_rows: Sequence[Mapping[str, Any]],
    out: Path,
) -> list[str]:
    plt = _pyplot()
    by_channel = {
        channel: _head_target(
            [row for row in causal_rows if row["channel"] == channel], "mediation"
        )
        for channel in ("semantic", "pe")
    }
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 12.0), constrained_layout=True)
    scatter = None
    methods = _available_methods(derived_rows)
    for ax, method in zip(axes.flat, methods):
        rows = [row for row in derived_rows if row["method"] == method]
        rows.sort(key=lambda row: (int(row["layer"]), int(row["head"])))
        d_rel = np.asarray([float(row["D_rel"]) for row in rows])
        differential = np.asarray(
            [
                by_channel["semantic"].get(
                    (int(row["layer"]), int(row["head"])), np.nan
                )
                - by_channel["pe"].get(
                    (int(row["layer"]), int(row["head"])), np.nan
                )
                for row in rows
            ]
        )
        layer = np.asarray([int(row["layer"]) for row in rows])
        current = _scatter(
            ax,
            d_rel,
            differential,
            layer,
            title=method,
            xlabel="Score role D_rel",
            ylabel="Held-out semantic − PE mediation",
            diagonal=False,
        )
        if current is not None:
            scatter = current
        ax.axvline(0.0, color="0.65", linewidth=0.8)
        ax.axhline(0.0, color="0.65", linewidth=0.8)
    _disable_unused(axes.flat, len(methods))
    if scatter is not None:
        fig.colorbar(scatter, ax=axes, label="Layer", shrink=0.65)
    paths = _save(fig, out / "role_validation")
    plt.close(fig)
    return paths


def topology_companions(rows: Sequence[Mapping[str, Any]], out: Path) -> list[str]:
    plt = _pyplot()
    fig, axes = plt.subplots(5, 4, figsize=(17.0, 19.0), constrained_layout=True)
    scatter = None
    flat_axes = list(axes.flat)
    methods = _available_methods(rows)
    for method_index, method in enumerate(methods):
        semantic, topology, layer = _method_arrays(
            rows, method, "semantic_score", "topology_score"
        )
        current = _scatter(
            flat_axes[2 * method_index],
            semantic,
            topology,
            layer,
            title=f"{method}: semantic companion",
            xlabel=RAW_AXIS_LABELS[method][0],
            ylabel="Fixed topology EG",
            diagonal=False,
        )
        if current is not None:
            scatter = current
        pe, topology, layer = _method_arrays(
            rows, method, "pe_score", "topology_score"
        )
        current = _scatter(
            flat_axes[2 * method_index + 1],
            pe,
            topology,
            layer,
            title=f"{method}: PE companion",
            xlabel=RAW_AXIS_LABELS[method][1],
            ylabel="Fixed topology EG",
            diagonal=False,
        )
        if current is not None:
            scatter = current
    for ax in flat_axes[2 * len(methods):]:
        ax.axis("off")
    if scatter is not None:
        fig.colorbar(scatter, ax=axes, label="Layer", shrink=0.65)
    paths = _save(fig, out / "topology_companions")
    plt.close(fig)
    return paths


def make_all_figures(
    raw_rows: Sequence[Mapping[str, Any]],
    derived_rows: Sequence[Mapping[str, Any]],
    *,
    out_dir: Path,
    ablation_rows: Sequence[Mapping[str, Any]] = (),
    causal_rows: Sequence[Mapping[str, Any]] = (),
) -> dict[str, list[str]]:
    """Render every required artifact; validation plots require cached validation tables."""

    output = {
        "raw_score_atlas": raw_score_atlas(raw_rows, out_dir),
        "m1_intervention_factorial": m1_factorial(raw_rows, out_dir),
        "donor_vs_transposition": donor_vs_transposition(raw_rows, out_dir),
        "derived_DJ_atlas": derived_dj_atlas(derived_rows, out_dir),
        "topology_companions": topology_companions(raw_rows, out_dir),
    }
    if ablation_rows:
        output["significance_validation"] = significance_validation(
            derived_rows, ablation_rows, out_dir
        )
    if causal_rows:
        output["role_validation"] = role_validation(
            derived_rows, causal_rows, out_dir
        )
    return output
