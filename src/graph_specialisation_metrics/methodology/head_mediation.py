"""Compare canonical internal head scores with finite output mediation.

This is a read-only analysis of completed canonical score and causal caches.  For
channel ``x`` and attention head ``h`` it compares:

* ``S_x(h)``: canonical output-projected donor-swap transport; and
* ``M_x(h)``: symmetric finite injection/restoration of the same head, measured
  by the canonical matched bidirectional gross patch response.

The two quantities use different native units, so the headline comparison is
their allocation across heads within each intervention channel.  No model,
dataset, checkpoint, or RRWP preprocessing is loaded in a figures-only run.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .cache import load_cache_artifact_file
from .grit_figure_data import CanonicalHeadMetrics, load_canonical_score_artifact


PROTOCOL_VERSION = "canonical-head-mediation-v1"
CHANNELS = ("semantic", "structural")
CHANNEL_COLOURS = {"semantic": "#0072B2", "structural": "#009E73"}
MEASURE_COLOURS = {"S": "#4D4D4D", "M": "#CC79A7"}


@dataclass(frozen=True)
class Config:
    canonical_root: Path
    output_dir: Path
    task: str = "zinc"
    train_seed: int = 42
    top_heads: int = 8
    bootstrap_replicates: int = 2_000
    analysis_seed: int = 91_733

    @property
    def task_root(self) -> Path:
        return self.canonical_root / self.task / f"seed_{int(self.train_seed)}"

    @property
    def score_path(self) -> Path:
        return self.task_root / "cache" / "scores" / "raw.pt"

    @property
    def causal_path(self) -> Path:
        return self.task_root / "cache" / "causal" / "validation.pt"


def _field(value: Any, name: str) -> Any:
    return value[name] if isinstance(value, Mapping) else getattr(value, name)


def _as_array(value: Any, *, dtype=float) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _head_name(layer: int, head: int) -> str:
    return f"head_L{int(layer)}_H{int(head)}"


def _display_head(layer: int, head: int) -> str:
    return f"L{int(layer) + 1} H{int(head) + 1}"


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    position = 0
    while position < values.size:
        end = position + 1
        while end < values.size and values[order[end]] == values[order[position]]:
            end += 1
        ranks[order[position:end]] = 0.5 * (position + end - 1) + 1.0
        position = end
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float).reshape(-1)
    right = np.asarray(right, dtype=float).reshape(-1)
    keep = np.isfinite(left) & np.isfinite(right)
    if int(keep.sum()) < 3:
        return float("nan")
    a, b = _rankdata(left[keep]), _rankdata(right[keep])
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _normalise_mass(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = np.where(np.isfinite(values), np.maximum(values, 0.0), 0.0)
    total = float(values.sum())
    return values / total if total > 0 else np.full_like(values, np.nan)


def _aggregate_graph_rows(rows: Sequence[Mapping[str, Any]], key: str) -> dict[int, float]:
    """Donors within source, then sources within graph."""

    graphs: dict[int, float] = {}
    graph_ids = sorted({int(row["graph"]) for row in rows})
    for graph in graph_ids:
        graph_rows = [row for row in rows if int(row["graph"]) == graph]
        source_values = []
        for source in sorted({int(row["source"]) for row in graph_rows}):
            values = [
                float(row[key])
                for row in graph_rows
                if int(row["source"]) == source and np.isfinite(float(row[key]))
            ]
            if values:
                source_values.append(float(np.mean(values)))
        if source_values:
            graphs[graph] = float(np.mean(source_values))
    return graphs


def _mediation_graph_matrix(
    causal: Mapping[str, Any],
    channel: str,
    shape: tuple[int, int],
) -> tuple[np.ndarray, tuple[int, ...]]:
    records = causal.get("event_records", {})
    per_head: list[dict[int, float]] = []
    for layer in range(shape[0]):
        for head in range(shape[1]):
            rows = records.get(_head_name(layer, head), {}).get(channel, ())
            per_head.append(_aggregate_graph_rows(rows, "P_gross_matched"))
    if not per_head or any(not values for values in per_head):
        return np.empty((0, shape[0] * shape[1])), ()
    available = [set(values) for values in per_head]
    common = sorted(set.intersection(*available))
    if not common:
        return np.empty((0, shape[0] * shape[1])), ()
    matrix = np.asarray(
        [[values[graph] for values in per_head] for graph in common],
        dtype=float,
    )
    return matrix, tuple(int(value) for value in common)


def _bootstrap_mediation_shares(
    graph_matrix: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if graph_matrix.shape[0] < 2 or int(replicates) < 2:
        point = _normalise_mass(np.mean(graph_matrix, axis=0))
        return point.copy(), point.copy()
    rng = np.random.default_rng(int(seed))
    draws = np.empty((int(replicates), graph_matrix.shape[1]), dtype=float)
    for draw in range(int(replicates)):
        selected = rng.integers(0, graph_matrix.shape[0], graph_matrix.shape[0])
        draws[draw] = _normalise_mass(np.mean(graph_matrix[selected], axis=0))
    return np.nanquantile(draws, 0.025, axis=0), np.nanquantile(draws, 0.975, axis=0)


def _contract_warnings(score_artifact: Any, causal_artifact: Any) -> list[str]:
    score = score_artifact.metadata.get("contract", {})
    causal = causal_artifact.metadata.get("contract", {})
    warnings: list[str] = []
    for field in (
        "task",
        "train_seed",
        "checkpoint_sha256",
        "task_adapter_version",
        "model_geometry",
        "split_fingerprint",
        "sigma",
    ):
        if score.get(field) != causal.get(field):
            warnings.append(
                f"score/causal contract mismatch for {field}: "
                f"{score.get(field)!r} != {causal.get(field)!r}"
            )
    return warnings


def analyse(config: Config) -> dict[str, Any]:
    score_artifact = load_canonical_score_artifact(
        config.score_path, expected_task=config.task
    )
    causal_artifact = load_cache_artifact_file(config.causal_path)
    warnings = _contract_warnings(score_artifact, causal_artifact)
    scores = score_artifact.value
    causal = causal_artifact.value
    metrics = CanonicalHeadMetrics.from_scores(scores)
    shape = metrics.shape
    n_heads = int(shape[0] * shape[1])

    score_interval = scores.get("intervals")
    score_low = _as_array(_field(score_interval, "low")) if score_interval is not None else None
    score_high = _as_array(_field(score_interval, "high")) if score_interval is not None else None
    summary_targets = causal.get("summary", {}).get("targets", {})

    channel_arrays: dict[str, dict[str, np.ndarray]] = {}
    rows: list[dict[str, Any]] = []
    channel_summary: list[dict[str, Any]] = []
    for channel_index, channel in enumerate(CHANNELS):
        s_raw = _as_array(
            metrics.raw_semantic if channel == "semantic" else metrics.raw_structural
        ).reshape(-1)
        s_share = _normalise_mass(s_raw)
        m_raw = np.asarray(
            [
                float(
                    summary_targets.get(_head_name(layer, head), {})
                    .get(channel, {})
                    .get("P_gross_matched", np.nan)
                )
                for layer in range(shape[0])
                for head in range(shape[1])
            ],
            dtype=float,
        )
        m_adjusted = np.asarray(
            [
                float(
                    summary_targets.get(_head_name(layer, head), {})
                    .get(channel, {})
                    .get("G_c", np.nan)
                )
                for layer in range(shape[0])
                for head in range(shape[1])
            ],
            dtype=float,
        )
        m_mismatch = np.asarray(
            [
                float(
                    summary_targets.get(_head_name(layer, head), {})
                    .get(channel, {})
                    .get("P_gross_mismatch", np.nan)
                )
                for layer in range(shape[0])
                for head in range(shape[1])
            ],
            dtype=float,
        )
        m_share = _normalise_mass(m_raw)
        if not np.isfinite(m_raw).all():
            warnings.append(
                f"{channel}: {int((~np.isfinite(m_raw)).sum())} head mediation estimates "
                "are unavailable and were assigned zero allocation"
            )

        if score_low is not None and score_low.shape[0] >= 4:
            normalized_position = 2 + channel_index
            # Canonical normalized scores have mean one; divide by head count for shares.
            s_low = _as_array(score_low[normalized_position]).reshape(-1) / n_heads
            s_high = _as_array(score_high[normalized_position]).reshape(-1) / n_heads
        else:
            s_low = s_share.copy()
            s_high = s_share.copy()
            warnings.append(f"{channel}: canonical score intervals unavailable")

        graph_matrix, graph_ids = _mediation_graph_matrix(causal, channel, shape)
        if graph_matrix.size:
            m_low, m_high = _bootstrap_mediation_shares(
                graph_matrix,
                replicates=config.bootstrap_replicates,
                seed=config.analysis_seed + channel_index,
            )
            reconstructed = np.mean(graph_matrix, axis=0)
            finite = np.isfinite(m_raw) & np.isfinite(reconstructed)
            relative = np.max(
                np.abs(m_raw[finite] - reconstructed[finite])
                / np.maximum(np.abs(m_raw[finite]), 1.0e-12)
            ) if finite.any() else np.nan
            if np.isfinite(relative) and relative > 1.0e-5:
                warnings.append(
                    f"{channel}: cached causal summary differs from event reconstruction "
                    f"(max relative error={relative:.3g})"
                )
        else:
            m_low = m_share.copy()
            m_high = m_share.copy()
            graph_ids = ()
            warnings.append(f"{channel}: event-level mediation records unavailable; no bootstrap")

        channel_arrays[channel] = {
            "S_raw": s_raw,
            "S_share": s_share,
            "S_low": s_low,
            "S_high": s_high,
            "M_raw": m_raw,
            "M_adjusted": m_adjusted,
            "M_mismatch": m_mismatch,
            "M_share": m_share,
            "M_low": m_low,
            "M_high": m_high,
        }
        k = min(int(config.top_heads), n_heads)
        top_s = set(np.argsort(-s_share)[:k].tolist())
        top_m = set(np.argsort(-m_share)[:k].tolist())
        channel_summary.append(
            {
                "channel": channel,
                "heads": n_heads,
                "causal_graphs": len(graph_ids),
                "spearman_raw": _spearman(s_raw, m_raw),
                "spearman_mismatch_adjusted": _spearman(s_raw, m_adjusted),
                "allocation_tv": float(0.5 * np.nansum(np.abs(s_share - m_share))),
                "top_k": k,
                "top_k_overlap": len(top_s & top_m),
                "effective_heads_S": float(1.0 / np.nansum(s_share**2)),
                "effective_heads_M": float(1.0 / np.nansum(m_share**2)),
            }
        )
        for flat, (layer, head) in enumerate(
            (layer, head) for layer in range(shape[0]) for head in range(shape[1])
        ):
            rows.append(
                {
                    "channel": channel,
                    "layer": int(layer),
                    "head": int(head),
                    "head_label": _display_head(layer, head),
                    "active_canonical": bool(metrics.active[layer, head]),
                    "S_raw": float(s_raw[flat]),
                    "S_share": float(s_share[flat]),
                    "S_share_low": float(s_low[flat]),
                    "S_share_high": float(s_high[flat]),
                    "M_raw": float(m_raw[flat]),
                    "M_mismatch_adjusted_raw": float(m_adjusted[flat]),
                    "M_mismatch_control_raw": float(m_mismatch[flat]),
                    "M_share": float(m_share[flat]),
                    "M_share_low": float(m_low[flat]),
                    "M_share_high": float(m_high[flat]),
                }
            )

    s_sem = channel_arrays["semantic"]["S_share"] * n_heads
    s_str = channel_arrays["structural"]["S_share"] * n_heads
    m_sem = channel_arrays["semantic"]["M_share"] * n_heads
    m_str = channel_arrays["structural"]["M_share"] * n_heads
    epsilon = 1.0e-12
    coordinates = {
        "S_J": 0.5 * (s_sem + s_str),
        "M_J": 0.5 * (m_sem + m_str),
        "S_D": (s_sem - s_str) / (s_sem + s_str + epsilon),
        "M_D": (m_sem - m_str) / (m_sem + m_str + epsilon),
    }
    coordinate_summary = {
        "joint_sensitivity_spearman": _spearman(coordinates["S_J"], coordinates["M_J"]),
        "selectivity_spearman": _spearman(coordinates["S_D"], coordinates["M_D"]),
    }
    return {
        "config": config,
        "shape": shape,
        "rows": rows,
        "channel_arrays": channel_arrays,
        "channel_summary": channel_summary,
        "coordinates": coordinates,
        "coordinate_summary": coordinate_summary,
        "warnings": warnings,
        "score_cache": str(config.score_path),
        "causal_cache": str(config.causal_path),
    }


def _layer_colours(layers: int) -> list[Any]:
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("viridis")
    return [cmap(value) for value in np.linspace(0.08, 0.90, max(int(layers), 2))]


def _identity_limits(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    maximum = float(np.nanmax(np.concatenate((left, right))))
    return 0.0, maximum * 1.08 if maximum > 0 else 1.0


def plot_headline(config: Config, analysis: Mapping[str, Any]) -> dict[str, str]:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    shape = tuple(analysis["shape"])
    layers = np.repeat(np.arange(shape[0]), shape[1])
    colours = _layer_colours(shape[0])
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.2))
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.09, top=0.82, hspace=0.43, wspace=0.33)
    for row_index, channel in enumerate(CHANNELS):
        values = analysis["channel_arrays"][channel]
        s_share = values["S_share"]
        m_share = values["M_share"]
        axis = axes[row_index, 0]
        for layer in range(shape[0]):
            mask = layers == layer
            axis.scatter(
                s_share[mask],
                m_share[mask],
                s=38,
                color=colours[layer],
                edgecolor="white",
                linewidth=0.6,
                alpha=0.90,
                zorder=2,
            )
        limits = _identity_limits(s_share, m_share)
        axis.plot(limits, limits, color="#999999", linestyle="--", linewidth=1.1, zorder=1)
        axis.set_xlim(limits)
        axis.set_ylim(limits)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel(r"Internal allocation $S_x(h)$")
        axis.set_ylabel(r"Finite mediation allocation $M_x(h)$")
        summary = next(row for row in analysis["channel_summary"] if row["channel"] == channel)
        axis.set_title(
            f"{channel.title()} heads: rank $\\rho$={summary['spearman_raw']:.2f}, "
            f"TV={summary['allocation_tv']:.2f}"
        )
        discrepancy = np.argsort(-np.abs(s_share - m_share))[:3]
        for flat in discrepancy:
            layer, head = divmod(int(flat), shape[1])
            axis.annotate(
                _display_head(layer, head),
                (s_share[flat], m_share[flat]),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=7.5,
                color="#444444",
            )

        axis = axes[row_index, 1]
        ordering = np.argsort(-np.maximum(s_share, m_share))[: min(config.top_heads, s_share.size)]
        ordering = ordering[np.argsort(np.maximum(s_share, m_share)[ordering])]
        positions = np.arange(ordering.size)
        labels = []
        for position, flat in enumerate(ordering):
            layer, head = divmod(int(flat), shape[1])
            labels.append(_display_head(layer, head))
            axis.plot(
                [s_share[flat], m_share[flat]],
                [position, position],
                color="#C8C8C8",
                linewidth=1.4,
                zorder=1,
            )
            for measure, marker in (("S", "^"), ("M", "o")):
                point = values[f"{measure}_share"][flat]
                low = values[f"{measure}_low"][flat]
                high = values[f"{measure}_high"][flat]
                error = np.asarray([[max(point - low, 0.0)], [max(high - point, 0.0)]])
                axis.errorbar(
                    point,
                    position,
                    xerr=error,
                    marker=marker,
                    markersize=6,
                    color=MEASURE_COLOURS[measure],
                    linestyle="none",
                    capsize=2,
                    zorder=2 if measure == "S" else 3,
                )
        axis.set_yticks(positions, labels)
        axis.set_xlim(left=0)
        axis.set_xlabel("Share of channel mass")
        axis.set_title(f"Largest {channel} contributors")
        axis.grid(axis="x", color="#E6E6E6", linewidth=0.8)

    for axis in axes.reshape(-1):
        axis.spines[["top", "right"]].set_visible(False)
    layer_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=colours[layer],
               markeredgecolor="white", label=f"Layer {layer + 1}")
        for layer in range(shape[0])
    ]
    measure_handles = [
        Line2D([0], [0], marker="^", linestyle="none", color=MEASURE_COLOURS["S"],
               label="Internal specialisation"),
        Line2D([0], [0], marker="o", linestyle="none", color=MEASURE_COLOURS["M"],
               label="Finite head mediation"),
    ]
    fig.legend(
        handles=layer_handles + measure_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.895),
        ncol=min(len(layer_handles) + 2, 7),
        frameon=False,
        fontsize=8.5,
    )
    fig.suptitle("Internal head specialisation and finite output mediation in Dense GRIT on ZINC", fontsize=15, y=0.975)
    fig.text(
        0.5,
        0.925,
        "Canonical transport versus matched symmetric injection/restoration; each is normalised across heads within channel",
        ha="center",
        color="#666666",
        fontsize=9,
    )
    figure_dir = config.output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    stem = figure_dir / "dense_zinc_head_specialisation_vs_mediation"
    png, pdf = stem.with_suffix(".png"), stem.with_suffix(".pdf")
    fig.savefig(png, dpi=260, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {"headline_png": str(png), "headline_pdf": str(pdf)}


def plot_coordinates(config: Config, analysis: Mapping[str, Any]) -> dict[str, str]:
    import matplotlib.pyplot as plt

    shape = tuple(analysis["shape"])
    layers = np.repeat(np.arange(shape[0]), shape[1])
    colours = _layer_colours(shape[0])
    coordinates = analysis["coordinates"]
    summary = analysis["coordinate_summary"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.35))
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.16, top=0.72, wspace=0.30)
    panels = (
        ("S_J", "M_J", r"Internal joint response $J_S$", r"Mediated joint response $J_M$", "Head response"),
        ("S_D", "M_D", r"Internal selectivity $D_S$", r"Mediated selectivity $D_M$", "Semantic--structural selectivity"),
    )
    for axis, (x_name, y_name, x_label, y_label, title) in zip(axes, panels, strict=True):
        x, y = coordinates[x_name], coordinates[y_name]
        for layer in range(shape[0]):
            mask = layers == layer
            axis.scatter(x[mask], y[mask], s=42, color=colours[layer], edgecolor="white", linewidth=0.6)
        if x_name == "S_D":
            axis.axhline(0, color="#BBBBBB", linewidth=0.9)
            axis.axvline(0, color="#BBBBBB", linewidth=0.9)
            axis.plot([-1, 1], [-1, 1], color="#888888", linestyle="--", linewidth=1)
            axis.set_xlim(-1.04, 1.04)
            axis.set_ylim(-1.04, 1.04)
        else:
            limits = _identity_limits(x, y)
            axis.plot(limits, limits, color="#888888", linestyle="--", linewidth=1)
            axis.set_xlim(limits)
            axis.set_ylim(limits)
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.set_title(title)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Internal and mediated head coordinates in Dense GRIT on ZINC", fontsize=15, y=0.96)
    fig.text(
        0.5,
        0.82,
        (
            f"Across heads: joint-response rank $\\rho$={summary['joint_sensitivity_spearman']:.2f}; "
            f"selectivity rank $\\rho$={summary['selectivity_spearman']:.2f}"
        ),
        ha="center",
        color="#666666",
        fontsize=9.5,
    )
    figure_dir = config.output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    stem = figure_dir / "dense_zinc_internal_vs_mediated_coordinates"
    png, pdf = stem.with_suffix(".png"), stem.with_suffix(".pdf")
    fig.savefig(png, dpi=260, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {"coordinates_png": str(png), "coordinates_pdf": str(pdf)}


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(config: Config) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        analysis = analyse(config)
    except Exception as error:  # Colab diagnostics are deliberately soft-fail.
        warning = f"{type(error).__name__}: {error}"
        print(f"[head-mediation:warning] {warning}")
        result = {
            "status": "not_estimable",
            "warnings": [warning],
            "score_cache": str(config.score_path),
            "causal_cache": str(config.causal_path),
            "figures": {},
        }
        (config.output_dir / "reported_error_audit.json").write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
        )
        return result

    figures = {}
    figures.update(plot_headline(config, analysis))
    figures.update(plot_coordinates(config, analysis))
    result_dir = config.output_dir / "results"
    _write_csv(result_dir / "head_channel_summary.csv", analysis["rows"])
    _write_csv(result_dir / "channel_summary.csv", analysis["channel_summary"])
    summary = {
        "status": "complete",
        "protocol_version": PROTOCOL_VERSION,
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "channel_summary": analysis["channel_summary"],
        "coordinate_summary": analysis["coordinate_summary"],
        "warnings": analysis["warnings"],
        "score_cache": analysis["score_cache"],
        "causal_cache": analysis["causal_cache"],
        "figures": figures,
    }
    (result_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    for warning in analysis["warnings"]:
        print(f"[head-mediation:warning] {warning}")
    for row in analysis["channel_summary"]:
        print(
            f"[{row['channel']}] rho={row['spearman_raw']:.3f} "
            f"TV={row['allocation_tv']:.3f} "
            f"top-{row['top_k']} overlap={row['top_k_overlap']}/{row['top_k']}"
        )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", default="zinc")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--top-heads", type=int, default=8)
    parser.add_argument("--bootstrap-replicates", type=int, default=2_000)
    parser.add_argument("--analysis-seed", type=int, default=91_733)
    args = parser.parse_args(argv)
    return Config(**vars(args))


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    return run(parse_args(argv))


if __name__ == "__main__":
    main()
