"""Compact paper figures for causal specialisation and effective transport.

This module deliberately does not depend on the legacy figure suite.  It consumes the
serialisable dictionaries returned by the optional causal extension and writes exactly two
headline figures per task, in both PNG and PDF form:

* channel-specific causal mediation; and
* receiver-specific routing versus effective relational transport.

The readers below accept both the canonical long-row schema (``summary_rows`` / ``rows``) and
small synthetic fixtures.  Missing optional estimates are rendered as an explicit unavailable or
inconclusive result rather than silently dropping a panel.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


SEMANTIC_BLUE = "#0072B2"
RRWP_ORANGE = "#D55E00"
NEUTRAL_GREY = "#777777"
LIGHT_GREY = "#D0D0D0"
DARK = "#222222"


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _first(mapping: Mapping[str, Any], names: Sequence[str], default: Any = None) -> Any:
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return default


def _finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _normalise_channel(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if any(token in text for token in ("rrwp", "struct", "role")):
        return "rrwp"
    if any(token in text for token in ("sem", "content", "atom")):
        return "semantic"
    if any(token in text for token in ("match", "random", "control", "null")):
        return "control"
    return text


def _normalise_direction(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if "denois" in text or "restor" in text or "rescue" in text:
        return "denoising"
    if "nois" in text or "damage" in text or "corrupt" in text:
        return "noising"
    return text


def _normalise_intervention(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    # Transport treatments are checked before channel aliases: e.g. ``remove_content_residual``
    # contains the word "content" but is an intervention, not a semantic counterfactual label.
    if text in {"static_routing", "routing_static", "static_attention", "staticise_routing"}:
        return "staticise_attention"
    if "remove_content_residual" in text:
        return "remove_content_residual"
    if "remove_edge_residual" in text:
        return "remove_edge_residual"
    if "residual_permut" in text:
        return "residual_permuted"
    if "attention" in text and any(token in text for token in ("static", "mean", "broadcast")):
        return "staticise_attention"
    if any(token in text for token in ("effective_static", "transport_static", "broadcast_only")):
        return "broadcast_only"
    if "remove" in text and "relational" in text:
        return "remove_relational"
    if "remove" in text and "broadcast" in text:
        return "remove_broadcast"
    if "relational_only" in text or "keep_relational" in text:
        return "relational_only"
    if any(token in text for token in ("zero", "full_ablat", "head_ablat")):
        return "head_zero"
    channel = _normalise_channel(text)
    if channel in {"semantic", "rrwp", "control"}:
        return channel
    if any(token in text for token in ("match", "random", "control", "null")):
        return "control"
    return text


def _row_effect(row: Mapping[str, Any]) -> float:
    return _finite_float(
        _first(
            row,
            (
                "estimate",
                "effect",
                "mean",
                "value",
                "mediation",
                "projected_mediation",
                "delta_mae",
                "loss_effect",
                "effect_mean",
                "beta",
                "double_dissociation",
                "interaction_beta",
            ),
        )
    )


def _row_ci(row: Mapping[str, Any]) -> tuple[float, float]:
    lo = _finite_float(_first(row, ("ci_low", "ci_lo", "lower", "lo", "effect_ci_low")))
    hi = _finite_float(_first(row, ("ci_high", "ci_hi", "upper", "hi", "effect_ci_high")))
    return lo, hi


def _row_k(row: Mapping[str, Any], default: int = 1) -> int:
    raw = _first(row, ("k", "topk", "top_k", "group_size"), default)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return int(default)


def _rows(container: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Read a long-row collection without imposing pandas on the analysis."""

    candidates = (
        "summary_rows",
        "rows",
        "mediation_rows",
        "causal_rows",
        "topk_rows",
        "intervention_rows",
    )
    for key in candidates:
        value = container.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, np.ndarray)):
            return [row for row in value if isinstance(row, Mapping)]
    groups = _as_mapping(container.get("groups"))
    value = groups.get("summary")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, np.ndarray)):
        return [row for row in value if isinstance(row, Mapping)]
    return []


def _bootstrap_mean_ci(
    values: Sequence[float], *, seed: int = 0, replicates: int = 2000
) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    estimate = float(arr.mean())
    if arr.size == 1:
        return estimate, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    reps = max(100, int(replicates))
    draws = rng.integers(0, arr.size, size=(reps, arr.size))
    boot = arr[draws].mean(axis=1)
    lo, hi = np.quantile(boot, (0.025, 0.975))
    return estimate, float(lo), float(hi)


def _aggregate_raw_rows(
    rows: Sequence[Mapping[str, Any]], *, seed: int = 0, replicates: int = 2000
) -> list[dict[str, Any]]:
    """Aggregate raw graph rows, clustering repeated interventions within graph first."""

    raw = [row for row in rows if np.isfinite(_row_effect(row))]
    if not raw:
        return []
    # Rows with explicit CIs/means are already summaries and should pass through unchanged.
    if any(np.isfinite(_row_ci(row)[0]) or np.isfinite(_row_ci(row)[1]) for row in raw):
        return [dict(row) for row in raw]

    graph_keys = ("graph_id", "graph", "gid", "molecule_id")
    has_graph = any(_first(row, graph_keys) is not None for row in raw)
    if not has_graph:
        return [dict(row) for row in raw]

    grouped: dict[tuple[Any, ...], dict[Any, list[float]]] = defaultdict(lambda: defaultdict(list))
    exemplars: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for row in raw:
        direction = _normalise_direction(_first(row, ("direction", "patch_direction", "mode")))
        group_type = _normalise_channel(_first(row, ("group_type", "control_type")))
        group = (
            "control"
            if group_type == "control"
            else _normalise_channel(
                _first(
                    row,
                    (
                        "head_group",
                        "selected_group",
                        "selection",
                        "selector",
                        "group",
                        "score_channel",
                    ),
                )
            )
        )
        intervention = _normalise_intervention(
            _first(
                row,
                (
                    "intervention",
                    "corruption",
                    "intervention_channel",
                    "counterfactual_channel",
                    "channel",
                ),
            )
        )
        metric = str(_first(row, ("metric", "effect_type", "outcome"), "effect"))
        control = bool(_first(row, ("is_control", "control"), False)) or group == "control"
        key = (direction, group, intervention, _row_k(row), metric, control)
        graph_id = _first(row, graph_keys, len(grouped[key]))
        grouped[key][graph_id].append(_row_effect(row))
        exemplars[key] = row

    out: list[dict[str, Any]] = []
    for index, (key, by_graph) in enumerate(sorted(grouped.items(), key=lambda item: str(item[0]))):
        values = [float(np.mean(vals)) for vals in by_graph.values()]
        estimate, lo, hi = _bootstrap_mean_ci(
            values, seed=seed + index, replicates=replicates
        )
        direction, group, intervention, k, metric, control = key
        row = dict(exemplars[key])
        row.update(
            {
                "direction": direction,
                "head_group": group,
                "intervention": intervention,
                "k": k,
                "metric": metric,
                "is_control": control,
                "estimate": estimate,
                "ci_low": lo,
                "ci_high": hi,
                "n_graphs": len(values),
            }
        )
        out.append(row)
    return out


def _score_coordinates(
    result: Mapping[str, Any], mediation: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    calibration = _as_mapping(mediation.get("calibration"))
    importance = _first(
        calibration,
        ("importance", "head_importance", "I", "importance_score"),
        _first(mediation, ("importance", "head_importance", "I", "importance_score")),
    )
    preference = _first(
        calibration,
        ("selectivity", "preference", "head_selectivity", "q", "channel_preference"),
        _first(
            mediation,
            ("selectivity", "preference", "head_selectivity", "q", "channel_preference"),
        ),
    )
    if importance is not None and preference is not None:
        return (
            np.asarray(importance, dtype=float).reshape(-1),
            np.asarray(preference, dtype=float).reshape(-1),
        )

    sem = np.asarray(_first(result, ("S_sem", "semantic_scores"), []), dtype=float)
    rrwp = np.asarray(_first(result, ("S_str", "S_rrwp", "rrwp_scores"), []), dtype=float)
    if sem.size == 0 or rrwp.size == 0 or sem.shape != rrwp.shape:
        return np.array([], dtype=float), np.array([], dtype=float)
    a = sem / (float(np.nanmean(sem)) + 1.0e-12)
    b = rrwp / (float(np.nanmean(rrwp)) + 1.0e-12)
    return (a + b).reshape(-1), ((a - b) / (a + b + 1.0e-12)).reshape(-1)


def _head_shape(result: Mapping[str, Any], count: int) -> tuple[int, int]:
    shape = np.asarray(_first(result, ("S_sem", "semantic_scores"), [])).shape
    if len(shape) == 2 and int(np.prod(shape)) == count:
        return int(shape[0]), int(shape[1])
    layers = int(_first(result, ("L", "n_layers"), 1) or 1)
    heads = int(_first(result, ("H", "n_heads"), max(1, count // layers)) or 1)
    return layers, heads


def _selected_heads(mediation: Mapping[str, Any]) -> dict[str, list[tuple[int, int]]]:
    calibration = _as_mapping(mediation.get("calibration"))
    raw = _first(
        mediation,
        ("selected_heads", "head_selection", "selections"),
        _first(calibration, ("rankings", "selected_heads"), {}),
    )
    result: dict[str, list[tuple[int, int]]] = {"semantic": [], "rrwp": []}
    if not isinstance(raw, Mapping):
        return result
    for name, values in raw.items():
        channel = _normalise_channel(name)
        if channel not in result:
            continue
        if isinstance(values, Mapping):
            values = _first(values, ("heads", "selected", "top_heads"), [])
        if isinstance(values, np.ndarray):
            values = values.tolist()
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            continue
        parsed: list[tuple[int, int]] = []
        for value in values:
            if isinstance(value, Mapping):
                layer = _first(value, ("layer", "l"))
                head = _first(value, ("head", "h"))
                if layer is not None and head is not None:
                    parsed.append((int(layer), int(head)))
            elif isinstance(value, Sequence) and len(value) >= 2:
                parsed.append((int(value[0]), int(value[1])))
        result[channel] = parsed
    return result


def _ci_status(estimate: float, lo: float, hi: float, *, noun: str = "effect") -> str:
    """Short, direction-neutral inference label used directly on headline panels."""

    if not np.isfinite(estimate):
        return f"{noun.capitalize()} unavailable"
    if not np.isfinite(lo) or not np.isfinite(hi):
        return "Uncertainty unavailable"
    if lo <= 0.0 <= hi:
        return f"No resolved {noun}"
    if hi < 0.0:
        return f"Resolved reversed {noun}"
    return f"Resolved positive {noun}"


def _double_dissociation_rows(mediation: Mapping[str, Any]) -> list[dict[str, Any]]:
    contrasts = _as_mapping(mediation.get("contrasts"))
    value = _first(
        mediation,
        ("double_dissociation", "double_dissociation_rows", "topk_double_dissociation"),
        _first(contrasts, ("topk", "double_dissociation")),
    )
    if isinstance(value, Mapping):
        out: list[dict[str, Any]] = []
        for direction, rows in value.items():
            if isinstance(rows, Mapping):
                rows = [rows]
            if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
                for row in rows:
                    if isinstance(row, Mapping):
                        item = dict(row)
                        item.setdefault("direction", direction)
                        out.append(item)
        return out
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, np.ndarray)):
        return [dict(row) for row in value if isinstance(row, Mapping)]
    return []


def _find_dd(
    dd_rows: Sequence[Mapping[str, Any]], direction: str, k: int
) -> tuple[float, float, float] | None:
    for row in dd_rows:
        if _normalise_direction(_first(row, ("direction", "patch_direction", "mode"))) != direction:
            continue
        if _row_k(row) != int(k):
            continue
        estimate = _row_effect(row)
        lo, hi = _row_ci(row)
        return estimate, lo, hi
    return None


def _primary_k(container: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> int:
    configured = _first(container, ("primary_topk", "primary_k", "paper_k"))
    if configured is None:
        contrasts = _as_mapping(container.get("contrasts"))
        primary = contrasts.get("primary")
        if isinstance(primary, Mapping):
            configured = _first(primary, ("k", "primary_k"))
        elif isinstance(primary, Sequence) and not isinstance(primary, (str, bytes, np.ndarray)):
            primary_rows = [row for row in primary if isinstance(row, Mapping)]
            if primary_rows:
                configured = _first(primary_rows[0], ("k", "primary_k"))
    if configured is not None:
        return int(configured)
    ks = sorted({_row_k(row) for row in rows})
    if 4 in ks:
        return 4
    return ks[len(ks) // 2] if ks else 1


def _cell_rows(
    rows: Sequence[Mapping[str, Any]], direction: str, k: int
) -> list[dict[str, Any]]:
    out = []
    for raw in rows:
        row = dict(raw)
        row_direction = _normalise_direction(_first(row, ("direction", "patch_direction", "mode")))
        if row_direction and row_direction != direction:
            continue
        if _row_k(row) != int(k):
            continue
        group_type = _normalise_channel(_first(row, ("group_type", "control_type")))
        group = (
            "control"
            if group_type == "control"
            else _normalise_channel(
                _first(
                    row,
                    (
                        "head_group",
                        "selected_group",
                        "selection",
                        "selector",
                        "group",
                        "score_channel",
                    ),
                )
            )
        )
        intervention = _normalise_intervention(
            _first(
                row,
                (
                    "intervention",
                    "corruption",
                    "intervention_channel",
                    "counterfactual_channel",
                    "channel",
                ),
            )
        )
        if group not in {"semantic", "rrwp", "control"}:
            continue
        if intervention not in {"semantic", "rrwp"}:
            continue
        row["_group"] = group
        row["_intervention"] = intervention
        out.append(row)
    return out


def _plot_discovery(
    ax: plt.Axes, result: Mapping[str, Any], mediation: Mapping[str, Any]
) -> None:
    importance, preference = _score_coordinates(result, mediation)
    if importance.size == 0 or preference.size != importance.size:
        _empty_panel(ax, "Discovery head scores unavailable")
        return
    finite = np.isfinite(importance) & np.isfinite(preference)
    importance, preference = importance[finite], preference[finite]
    if importance.size == 0:
        _empty_panel(ax, "Discovery head scores unavailable")
        return

    colors = np.where(preference >= 0.0, SEMANTIC_BLUE, RRWP_ORANGE)
    ax.scatter(preference, importance, c=colors, s=28, alpha=0.72, edgecolors="none")
    ax.axvline(0.0, color=NEUTRAL_GREY, lw=0.8)
    ax.set_xlabel("channel preference  q\nRRWP-role  ←   →  semantic")
    ax.set_ylabel("importance  I")
    ax.set_title("a  Discovery selection", loc="left", fontweight="bold")

    selected = _selected_heads(mediation)
    layers, heads = _head_shape(result, int(finite.size))
    original_indices = np.flatnonzero(finite)
    reverse = {int(original): pos for pos, original in enumerate(original_indices)}
    for channel, head_list in selected.items():
        color = SEMANTIC_BLUE if channel == "semantic" else RRWP_ORANGE
        for layer, head in head_list[:4]:
            flat = int(layer) * heads + int(head)
            if flat not in reverse:
                continue
            pos = reverse[flat]
            ax.scatter(
                [preference[pos]], [importance[pos]], s=66, facecolors="none",
                edgecolors=color, linewidths=1.4, zorder=4,
            )
            ax.annotate(
                f"L{layer}H{head}", (preference[pos], importance[pos]), xytext=(3, 3),
                textcoords="offset points", fontsize=6.8, color=DARK,
            )


def _plot_mediation_panel(
    ax: plt.Axes,
    mediation: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    dd_rows: Sequence[Mapping[str, Any]],
    direction: str,
    k: int,
) -> None:
    cells = _cell_rows(rows, direction, k)
    if cells:
        x_of = {"semantic": 0.0, "rrwp": 1.0}
        styles = {
            "semantic": (SEMANTIC_BLUE, "o", "semantic-selected"),
            "rrwp": (RRWP_ORANGE, "s", "RRWP-role-selected"),
            "control": (NEUTRAL_GREY, "^", "matched control"),
        }
        for group in ("semantic", "rrwp", "control"):
            group_rows = [row for row in cells if row["_group"] == group]
            if not group_rows:
                continue
            color, marker, label = styles[group]
            points = []
            for channel in ("semantic", "rrwp"):
                candidates = [row for row in group_rows if row["_intervention"] == channel]
                if not candidates:
                    continue
                estimates = np.asarray([_row_effect(row) for row in candidates], dtype=float)
                estimates = estimates[np.isfinite(estimates)]
                if estimates.size == 0:
                    continue
                estimate = float(estimates.mean())
                lo, hi = (
                    _row_ci(candidates[0])
                    if len(candidates) == 1
                    else (float("nan"), float("nan"))
                )
                x = x_of[channel]
                yerr = None
                if np.isfinite(lo) and np.isfinite(hi):
                    yerr = [[max(0.0, estimate - lo)], [max(0.0, hi - estimate)]]
                ax.errorbar(
                    [x], [estimate], yerr=yerr, fmt=marker, color=color,
                    markersize=5.2, capsize=2.5, lw=1.1, label=label if not points else None,
                )
                points.append((x, estimate))
            if len(points) == 2:
                ax.plot(
                    [point[0] for point in points], [point[1] for point in points],
                    color=color, lw=1.1, alpha=0.8,
                )
        ax.set_xticks([0, 1], ["semantic", "RRWP-role"])
        ax.set_xlabel("counterfactual channel")
        ax.set_ylabel("projected mediation")
        ax.axhline(0.0, color=NEUTRAL_GREY, lw=0.7)
        if direction == "noising":
            ax.legend(frameon=False, fontsize=6.8, loc="best")
    else:
        # A top-k double-dissociation curve is the preferred fallback when cell summaries are not
        # retained.  It still exposes the held-out causal claim and its uncertainty.
        curve = [
            row for row in dd_rows
            if _normalise_direction(
                _first(row, ("direction", "patch_direction", "mode"))
            ) == direction
        ]
        if not curve:
            _empty_panel(ax, f"Held-out {direction} estimates unavailable")
            ax.set_title(
                ("b" if direction == "noising" else "c") + f"  Held-out {direction}",
                loc="left", fontweight="bold",
            )
            return
        curve = sorted(curve, key=_row_k)
        xs = np.array([_row_k(row) for row in curve], dtype=float)
        ys = np.array([_row_effect(row) for row in curve], dtype=float)
        los = np.array([_row_ci(row)[0] for row in curve], dtype=float)
        his = np.array([_row_ci(row)[1] for row in curve], dtype=float)
        ax.plot(xs, ys, "-o", color=DARK, lw=1.3, ms=4)
        if np.isfinite(los).all() and np.isfinite(his).all():
            ax.fill_between(xs, los, his, color=LIGHT_GREY, alpha=0.55, linewidth=0)
        control_mean = np.asarray(
            [_finite_float(row.get("matched_control_mean")) for row in curve], dtype=float
        )
        if np.isfinite(control_mean).any():
            ax.plot(xs, control_mean, "--", color=NEUTRAL_GREY, lw=1.0, label="matched control")
            control_lo = np.asarray(
                [_finite_float(row.get("matched_control_ci_low")) for row in curve], dtype=float
            )
            control_hi = np.asarray(
                [_finite_float(row.get("matched_control_ci_high")) for row in curve], dtype=float
            )
            if np.isfinite(control_lo).all() and np.isfinite(control_hi).all():
                ax.fill_between(
                    xs, control_lo, control_hi, color=LIGHT_GREY, alpha=0.28, linewidth=0
                )
            ax.legend(frameon=False, fontsize=6.8, loc="best")
        ax.axhline(0.0, color=NEUTRAL_GREY, lw=0.7)
        ax.set_xticks(xs)
        ax.set_xlabel("selected heads  k")
        ax.set_ylabel("double dissociation")

    subtitle_parts: list[str] = []
    dd = _find_dd(dd_rows, direction, k)
    if dd is not None:
        estimate, lo, hi = dd
        detail = f"DD={estimate:+.2f}"
        if np.isfinite(lo) and np.isfinite(hi):
            detail += f" [{lo:+.2f},{hi:+.2f}]"
            if lo <= 0.0 <= hi:
                detail += " unresolved (CI crosses 0)"
            elif hi < 0.0:
                detail += " reversed"
            else:
                detail += " resolved"
        else:
            detail += " (CI unavailable)"
        subtitle_parts.append(detail)

    panel = "b" if direction == "noising" else "c"
    subtitle = "\n" + " · ".join(subtitle_parts) if subtitle_parts else ""
    ax.set_title(
        f"{panel}  Held-out {direction}  (k={k}){subtitle}",
        loc="left", fontweight="bold", fontsize=7.5,
    )


def _empty_panel(ax: plt.Axes, message: str) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center", va="center", color=NEUTRAL_GREY)


def _figure_mediation(
    task: str, result: Mapping[str, Any], mediation: Mapping[str, Any]
) -> plt.Figure:
    raw_rows = _rows(mediation)
    rows = _aggregate_raw_rows(
        raw_rows,
        seed=int(_first(mediation, ("bootstrap_seed", "analysis_seed"), 0)),
        replicates=int(_first(mediation, ("bootstrap_replicates", "bootstrap_samples"), 2000)),
    )
    dd_rows = _double_dissociation_rows(mediation)
    k = _primary_k(mediation, rows + dd_rows)

    with plt.rc_context(_paper_rc()):
        fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.45), constrained_layout=True)
        _plot_discovery(axes[0], result, mediation)
        _plot_mediation_panel(axes[1], mediation, rows, dd_rows, "noising", k)
        _plot_mediation_panel(axes[2], mediation, rows, dd_rows, "denoising", k)
        title = str(_first(result, ("title",), task))
        fig.suptitle(
            f"{title} · channel-specific causal mediation", fontsize=11, fontweight="bold"
        )
    return fig


def _array_from(container: Mapping[str, Any], names: Sequence[str]) -> np.ndarray:
    value = _first(container, names)
    if value is None:
        return np.array([], dtype=float)
    try:
        return np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return np.array([], dtype=float)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Dependency-free average ranks for a descriptive Spearman coefficient."""

    values = np.asarray(values, dtype=float).reshape(-1)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    x_rank = _average_ranks(x)
    y_rank = _average_ranks(y)
    if float(np.std(x_rank)) == 0.0 or float(np.std(y_rank)) == 0.0:
        return float("nan")
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def _reconstruction_state(transport: Mapping[str, Any]) -> tuple[bool | None, float]:
    nested = _as_mapping(_first(transport, ("reconstruction", "reconstruction_check"), {}))
    checks = _as_mapping(transport.get("checks"))
    valid_raw = _first(
        nested,
        ("valid", "passed", "ok"),
        _first(
            checks,
            ("passed", "valid", "ok"),
            _first(transport, ("reconstruction_valid", "reconstruction_ok")),
        ),
    )
    valid = None if valid_raw is None else bool(valid_raw)
    error = _finite_float(
        _first(
            nested,
            ("relative_error", "max_relative_error", "error", "max_error"),
            _first(
                checks,
                (
                    "max_wv_reconstruction_error",
                    "max_oh_reconstruction_error",
                    "max_reconstruction_error",
                ),
                _first(
                    transport,
                    (
                        "reconstruction_error",
                        "reconstruction_relative_error",
                        "max_reconstruction_error",
                    ),
                ),
            ),
        )
    )
    return valid, error


def _transport_arrays(transport: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, str]:
    metrics = _as_mapping(transport.get("head_metrics"))
    source = metrics or transport
    receiver_dependence = _array_from(
        source,
        (
            "routing_receiver_dependence",
            "routing_relational_fraction",
            "attention_receiver_dependence",
            "routing_projected_relational_fraction",
        ),
    )
    if receiver_dependence.size:
        x = receiver_dependence
        xlabel = "routing receiver dependence"
    else:
        # Jensen--Shannon divergence is a distance *from* the support- and
        # source-marginal-matched static null.  Larger values therefore mean
        # more receiver-specific routing, not greater staticity.
        x = _array_from(source, ("routing_js_static",))
        if x.size:
            xlabel = "receiver-specific routing\n(JS from matched static null)"
        else:
            x = _array_from(
                source,
                (
                    "routing_staticity",
                    "attention_staticity",
                    "staticity",
                ),
            )
            xlabel = "routing staticity"
    y = _array_from(
        source,
        (
            "effective_relational_fraction",
            "effective_projected_relational_fraction",
            "transport_relational_fraction",
            "relational_fraction",
        ),
    )
    return x, y, xlabel


def _plot_transport_scatter(
    ax: plt.Axes,
    result: Mapping[str, Any],
    mediation: Mapping[str, Any],
    transport: Mapping[str, Any],
) -> None:
    x, y, xlabel = _transport_arrays(transport)
    if x.size == 0 or x.size != y.size:
        _empty_panel(ax, "Routing/transport decomposition unavailable")
        ax.set_title("a  Selection versus transport", loc="left", fontweight="bold")
        return
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size == 0:
        _empty_panel(ax, "Routing/transport decomposition unavailable")
        return

    ax.scatter(x, y, color=NEUTRAL_GREY, s=27, alpha=0.72, edgecolors="none")
    selected = _selected_heads(mediation)
    if not selected["semantic"] and not selected["rrwp"]:
        selected = _selected_heads(transport)
    _, heads = _head_shape(result, int(finite.size))
    original_indices = np.flatnonzero(finite)
    reverse = {int(original): pos for pos, original in enumerate(original_indices)}
    for channel, head_list in selected.items():
        color = SEMANTIC_BLUE if channel == "semantic" else RRWP_ORANGE
        for layer, head in head_list[:4]:
            flat = int(layer) * heads + int(head)
            if flat not in reverse:
                continue
            pos = reverse[flat]
            ax.scatter(
                [x[pos]], [y[pos]], color=color, s=42, edgecolors=DARK,
                linewidths=0.35, zorder=3,
            )
            ax.annotate(
                f"L{layer}H{head}", (x[pos], y[pos]), xytext=(3, 3), textcoords="offset points",
                fontsize=6.8,
            )
    rho = _spearman_rho(x, y)
    if np.isfinite(rho):
        ax.text(
            0.03,
            0.97,
            f"all-head Spearman ρ={rho:+.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.8,
            color=DARK,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("effective projected relational fraction")
    ax.set_title("a  Selection versus transport", loc="left", fontweight="bold")


def _transport_rows(transport: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_rows = _rows(transport)
    causal = _as_mapping(transport.get("causal"))
    if not raw_rows and causal:
        flattened: list[dict[str, Any]] = []
        for group_name, by_k_value in causal.items():
            by_k = _as_mapping(by_k_value)
            for k_value, by_treatment_value in by_k.items():
                by_treatment = _as_mapping(by_treatment_value)
                for treatment, payload_value in by_treatment.items():
                    payload = _as_mapping(payload_value)
                    summary = _as_mapping(
                        _first(payload, ("delta_mae", "loss_effect", "effect_summary"), {})
                    )
                    estimate = _finite_float(
                        _first(
                            summary,
                            ("mean", "estimate", "effect"),
                            _first(payload, ("mean", "estimate")),
                        )
                    )
                    lo = _finite_float(_first(summary, ("ci_low", "ci_lo", "lower")))
                    hi = _finite_float(_first(summary, ("ci_high", "ci_hi", "upper")))
                    flattened.append(
                        {
                            "group": group_name,
                            "k": int(k_value),
                            "intervention": treatment,
                            "estimate": estimate,
                            "ci_low": lo,
                            "ci_high": hi,
                            "n_graphs": _first(summary, ("n_graphs", "n")),
                        }
                    )
        raw_rows = flattened
    rows = _aggregate_raw_rows(
        raw_rows,
        seed=int(_first(transport, ("bootstrap_seed", "analysis_seed"), 0)),
        replicates=int(_first(transport, ("bootstrap_replicates", "bootstrap_samples"), 2000)),
    )
    out = []
    for raw in rows:
        row = dict(raw)
        intervention = _normalise_intervention(
            _first(row, ("intervention", "condition", "component", "ablation", "mode"))
        )
        row["_intervention"] = intervention
        row["_group"] = _normalise_channel(
            _first(row, ("group", "head_group", "selector", "selection", "selected_group"))
        )
        if np.isfinite(_row_effect(row)):
            out.append(row)
    return out


_TRANSPORT_STYLE = {
    "staticise_attention": (SEMANTIC_BLUE, "o", "staticise routing"),
    "broadcast_only": (RRWP_ORANGE, "s", "broadcast only"),
    "relational_only": (SEMANTIC_BLUE, "D", "relational only"),
    "remove_relational": (RRWP_ORANGE, "s", "remove relational"),
    "remove_broadcast": (SEMANTIC_BLUE, "D", "remove broadcast"),
    "residual_permuted": (RRWP_ORANGE, "P", "permute relational residual"),
    "remove_content_residual": (SEMANTIC_BLUE, "v", "remove content residual"),
    "remove_edge_residual": (RRWP_ORANGE, "^", "remove edge residual"),
    "head_zero": (DARK, "X", "zero selected heads"),
    "control": (NEUTRAL_GREY, "^", "matched control"),
}


def _plot_transport_topk(ax: plt.Axes, rows: Sequence[Mapping[str, Any]]) -> None:
    by_intervention: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_intervention[str(row.get("_intervention", ""))].append(row)
    candidates = [
        key for key, values in by_intervention.items()
        if len({_row_k(row) for row in values}) >= 2
    ]
    priority = (
        "broadcast_only",
        "staticise_attention",
        "residual_permuted",
        "remove_edge_residual",
        "remove_content_residual",
        "head_zero",
    )
    intervention = next(
        (key for key in priority if key in candidates),
        candidates[0] if candidates else "",
    )
    if not intervention:
        _empty_panel(ax, "Held-out top-k intervention curves unavailable")
        ax.set_title("b  Causal top-k test", loc="left", fontweight="bold")
        return
    values_by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in by_intervention[intervention]:
        values_by_group[str(row.get("_group", ""))].append(row)
    group_styles = {
        "semantic": (SEMANTIC_BLUE, "o", "semantic-selected"),
        "rrwp": (RRWP_ORANGE, "s", "RRWP-role-selected"),
        "control": (NEUTRAL_GREY, "^", "matched control"),
        "": (DARK, "o", "selected heads"),
    }
    plotted_ks: set[int] = set()
    for group, values in values_by_group.items():
        values = sorted(values, key=_row_k)
        color, marker, label = group_styles.get(group, (NEUTRAL_GREY, "o", group.replace("_", " ")))
        xs = np.asarray([_row_k(row) for row in values], dtype=float)
        ys = np.asarray([_row_effect(row) for row in values], dtype=float)
        los = np.asarray([_row_ci(row)[0] for row in values], dtype=float)
        his = np.asarray([_row_ci(row)[1] for row in values], dtype=float)
        ax.plot(xs, ys, marker=marker, color=color, lw=1.25, ms=4, label=label)
        if np.isfinite(los).all() and np.isfinite(his).all():
            ax.fill_between(xs, los, his, color=color, alpha=0.12, linewidth=0)
        plotted_ks.update(int(value) for value in xs)
    ax.axhline(0.0, color=NEUTRAL_GREY, lw=0.7)
    ks = sorted(plotted_ks)
    ax.set_xticks(ks)
    ax.set_xlabel("selected heads  k")
    ax.set_ylabel("ΔMAE after intervention\n(higher = worse)")
    treatment_label = _TRANSPORT_STYLE.get(
        intervention, (None, None, intervention.replace("_", " "))
    )[2]
    ax.set_title(f"b  Top-k: {treatment_label}", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=6.8, loc="best")


def _plot_transport_primary(
    ax: plt.Axes, transport: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> None:
    k = _primary_k(transport, rows)
    selected = [row for row in rows if _row_k(row) == k]
    unique: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in selected:
        unique.setdefault((str(row.get("_intervention", "")), str(row.get("_group", ""))), row)
    order = [
        key for key in (
            "staticise_attention", "broadcast_only", "relational_only",
            "residual_permuted", "remove_broadcast", "remove_relational",
            "remove_content_residual", "remove_edge_residual", "head_zero", "control",
        )
        if any(intervention == key for intervention, _group in unique)
    ]
    if not order:
        _empty_panel(ax, "Primary intervention effects unavailable")
        ax.set_title(f"c  Intervention effects  (k={k})", loc="left", fontweight="bold")
        return
    group_styles = {
        "semantic": (SEMANTIC_BLUE, "o", "semantic-selected"),
        "rrwp": (RRWP_ORANGE, "s", "RRWP-role-selected"),
        "control": (NEUTRAL_GREY, "^", "matched control"),
        "": (DARK, "o", "selected heads"),
    }
    offsets = {"semantic": -0.18, "rrwp": 0.18, "control": 0.0, "": 0.0}
    seen_groups: set[str] = set()
    inferential: list[Mapping[str, Any]] = []
    for index, intervention in enumerate(order):
        for group in ("semantic", "rrwp", "control", ""):
            row = unique.get((intervention, group))
            if row is None:
                continue
            estimate = _row_effect(row)
            lo, hi = _row_ci(row)
            color, marker, group_label = group_styles[group]
            xerr = None
            if np.isfinite(lo) and np.isfinite(hi):
                xerr = [[max(0.0, estimate - lo)], [max(0.0, hi - estimate)]]
            ax.errorbar(
                [estimate], [index + offsets[group]], xerr=xerr, fmt=marker, color=color,
                capsize=2.5, markersize=4.5, lw=1.0,
                label=group_label if group not in seen_groups else None,
            )
            seen_groups.add(group)
            if intervention not in {"head_zero", "control"} and group != "control":
                inferential.append(row)
    ax.axvline(0.0, color=NEUTRAL_GREY, lw=0.7)
    labels = [_TRANSPORT_STYLE.get(key, (None, None, key))[2] for key in order]
    ax.set_yticks(range(len(order)), labels)
    ax.invert_yaxis()
    ax.set_xlabel("ΔMAE after intervention  (higher = worse)")
    if seen_groups:
        ax.legend(frameon=False, fontsize=6.5, loc="best")
    subtitle = ""
    if inferential:
        resolved = [
            row for row in inferential
            if np.isfinite(_row_ci(row)[0])
            and np.isfinite(_row_ci(row)[1])
            and not (_row_ci(row)[0] <= 0.0 <= _row_ci(row)[1])
        ]
        status = (
            "No resolved component effect" if not resolved
            else f"{len(resolved)}/{len(inferential)} component effects exclude zero"
        )
        subtitle = f"\n{status.lower()}"
    ax.set_title(
        f"c  Intervention effects  (k={k}){subtitle}",
        loc="left", fontweight="bold", fontsize=8.2,
    )


def _figure_transport(
    task: str,
    result: Mapping[str, Any],
    mediation: Mapping[str, Any],
    transport: Mapping[str, Any],
) -> plt.Figure:
    rows = _transport_rows(transport)
    valid, error = _reconstruction_state(transport)
    with plt.rc_context(_paper_rc()):
        fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.45), constrained_layout=True)
        title = str(_first(result, ("title",), task))
        fig.suptitle(
            f"{title} · receiver-specific routing versus effective relational transport",
            fontsize=11, fontweight="bold",
        )
        if valid is False:
            # Never display downstream mechanism estimates after their defining
            # reconstruction invariant failed.  The failure itself is the result.
            failure_titles = (
                "a  Exact reconstruction",
                "b  Causal top-k test",
                "c  Intervention effects",
            )
            for ax, panel_title in zip(axes, failure_titles):
                _empty_panel(ax, "Analysis blocked\nexact pair→wV reconstruction failed")
                ax.set_title(panel_title, loc="left", fontweight="bold")
            checks = _as_mapping(transport.get("checks"))
            failure_message = str(_first(checks, ("failure_message", "error_message"), "")).strip()
            detail = "Mechanistic estimates withheld because transport validation failed"
            if np.isfinite(error):
                detail += f"  (reported max error {error:.2e})"
            elif failure_message:
                compact = " ".join(failure_message.split())
                detail += f": {compact[:150]}" + ("…" if len(compact) > 150 else "")
            fig.text(0.5, 0.005, detail, ha="center", va="bottom", fontsize=7.3, color=DARK)
        else:
            _plot_transport_scatter(axes[0], result, mediation, transport)
            _plot_transport_topk(axes[1], rows)
            _plot_transport_primary(axes[2], transport, rows)
            if valid is None:
                fig.text(
                    0.5,
                    0.005,
                    "pair→wV reconstruction not reported; interpretation is provisional",
                    ha="center", va="bottom", fontsize=7.0, color=NEUTRAL_GREY,
                )
    return fig


def _paper_rc() -> dict[str, Any]:
    return {
        "font.size": 8.3,
        "axes.titlesize": 8.8,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "axes.grid": True,
        "grid.alpha": 0.16,
        "grid.linewidth": 0.55,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.fontsize": 7.0,
        "savefig.facecolor": "white",
    }


def _save_pair(fig: plt.Figure, stem: Path) -> dict[str, str]:
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png), "pdf": str(pdf)}


def make_paper_figures(
    results: Mapping[str, Mapping[str, Any]],
    mediations: Mapping[str, Mapping[str, Any]],
    transports: Mapping[str, Mapping[str, Any]],
    out_dir: str | Path,
) -> dict[str, dict[str, dict[str, str]]]:
    """Write exactly two headline figure stems per task, each as PNG and PDF.

    Args:
        results: Existing per-task specialisation results. ``S_sem`` and ``S_str`` are sufficient
            to derive discovery importance/preference when the mediation result does not persist
            them directly.
        mediations: Per-task causal mediation outputs. Canonical keys are ``importance``,
            ``selectivity``, ``selected_heads``, ``summary_rows`` and ``double_dissociation``.
        transports: Per-task effective-transport outputs. Canonical keys are
            ``routing_receiver_dependence`` (or ``routing_staticity``),
            ``effective_relational_fraction``, ``summary_rows`` and ``reconstruction``.
        out_dir: Analysis output root. Files are written under ``out_dir / 'paper'``.

    Returns:
        ``{task: {figure_name: {'png': path, 'pdf': path}}}``.
    """

    paper_dir = Path(out_dir) / "paper"
    paper_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, dict[str, dict[str, str]]] = {}
    for task, result_value in results.items():
        result = _as_mapping(result_value)
        mediation = _as_mapping(mediations.get(task, {}))
        transport = _as_mapping(transports.get(task, {}))

        mediation_name = f"fig_channel_causal_mediation_{task}"
        transport_name = f"fig_effective_relational_transport_{task}"
        written[task] = {
            "channel_causal_mediation": _save_pair(
                _figure_mediation(task, result, mediation), paper_dir / mediation_name
            ),
            "effective_relational_transport": _save_pair(
                _figure_transport(task, result, mediation, transport), paper_dir / transport_name
            ),
        }
    return written


__all__ = ["make_paper_figures"]
