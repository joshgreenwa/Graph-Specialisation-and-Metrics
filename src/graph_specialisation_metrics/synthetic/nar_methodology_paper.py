"""Paper-facing, cache-only synthesis of the fixed-N NAR methodology experiment.

This module deliberately does not own an inference cache.  It opens the canonical score, causal,
and carriage artifacts plus the v1 exact-counterfactual extension as immutable inputs, derives
paper tables from their retained sufficient statistics, and writes a new versioned figure tree.

The scientific hierarchy follows the normative repository README:

1. show the mandatory structural-versus-semantic and D_rel-versus-J head landscapes;
2. validate raw scores and derived coordinates against held-out causal endpoints;
3. interpret the frozen core families through input-defined source-role strata;
4. treat address/content counterfactual mediation as a semantic subrole analysis; and
5. compare causal grounding, rather than a task-specific normalised score, across capacity states.

No function in this module loads a checkpoint or imports the official GRIT runtime.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..methodology.bootstrap import (
    Observation,
    nested_percentile_interval,
    trimmed_mean,
)
from ..methodology.cache import StaleCacheError, atomic_json, load_cache_artifact_file
from ..methodology.distance import display_bins
from ..methodology.protocol import BootstrapPolicy
from ..methodology.scores import aggregate_event_scores
from .nar_canonical_analysis import (
    MODEL_COLOURS,
    MODEL_LABELS,
    MODEL_MARKERS,
    MODEL_ORDER,
    SEED_MARKERS,
    _save_figure,
    _style,
    _write_csv,
    best_seed_by_validation,
    task_name,
)
from .nar_methodology_extension import (
    DEFAULT_EXTENSION_NAME as SOURCE_EXTENSION_NAME,
    ExtensionStore,
    _methodology_policies,
    _read_performance_rows,
    _safe_extension_layout,
    load_cached_counterfactual_results,
    load_cached_role_results,
    load_source_binding,
)


PAPER_VERSION = "nar-methodology-paper-v2"
DEFAULT_PAPER_ANALYSIS_NAME = "nar_methodology_paper_v2"

CORE_FIGURE_STEM = "01_core_specialisation_N{N}"
CAUSAL_FIGURE_STEM = "02_core_causal_validation_N{N}"
INTERPRETATION_FIGURE_STEM = "03_task_role_and_counterfactual_validation"
TRANSITION_FIGURE_STEM = "04_competence_and_causal_grounding"
CARRIAGE_FIGURE_STEM = "S02_role_conditioned_functional_carriage_N{N}"

LAYER_COLOURS = (
    "#1F7A8C",
    "#D18B35",
    "#76528B",
    "#5F6B6D",
)
SEMANTIC_COLOUR = "#B33A3A"
STRUCTURAL_COLOUR = "#2F6B9A"
CONTROL_COLOUR = "#8A8A8A"
INACTIVE_COLOUR = "#C7C9CC"

CORE_FAMILIES = ("semantic_leaning", "structural_leaning")
CONDITIONAL_COMPONENTS = (
    "semantic_query",
    "semantic_record",
    "structural_query",
    "structural_record",
)
CONDITIONAL_LABELS = (
    "Semantic\nquery",
    "Semantic\nrecord",
    "Structural\nquery",
    "Structural\nrecord",
)


def _parse_csv_strings(value: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        result = tuple(item.strip() for item in value.split(",") if item.strip())
    else:
        result = tuple(str(item) for item in value)
    if not result:
        raise ValueError("expected at least one string")
    return result


def _parse_csv_ints(value: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    else:
        result = tuple(int(item) for item in value)
    if not result:
        raise ValueError("expected at least one integer")
    return result


def _as_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _interval_arrays(interval: Any) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(interval, "low") and hasattr(interval, "high"):
        return (
            np.asarray(interval.low, dtype=np.float64),
            np.asarray(interval.high, dtype=np.float64),
        )
    if isinstance(interval, Mapping):
        return (
            np.asarray(interval["low"], dtype=np.float64),
            np.asarray(interval["high"], dtype=np.float64),
        )
    raise TypeError(f"unsupported interval payload {type(interval)!r}")


def _accuracy_values(
    rows: Sequence[Mapping[str, Any]],
    model: str,
    records: int,
    seeds: Sequence[int],
) -> np.ndarray:
    lookup = {
        (str(row.get("model")), int(row.get("N", -1)), int(row.get("seed", -1))): _as_float(
            row.get("accuracy")
        )
        for row in rows
    }
    values = np.asarray(
        [lookup.get((str(model), int(records), int(seed)), np.nan) for seed in seeds],
        dtype=np.float64,
    )
    return values[np.isfinite(values)]


def accuracy_summary(
    rows: Sequence[Mapping[str, Any]],
    model: str,
    records: int,
    seeds: Sequence[int],
) -> tuple[float, float, int]:
    values = _accuracy_values(rows, model, records, seeds)
    if not len(values):
        return float("nan"), float("nan"), 0
    return (
        float(np.mean(values)),
        float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        int(len(values)),
    )


def _accuracy_subtitle(
    rows: Sequence[Mapping[str, Any]],
    model: str,
    records: int,
    seeds: Sequence[int],
) -> str:
    mean, sd, count = accuracy_summary(rows, model, records, seeds)
    if not np.isfinite(mean):
        return "held-out accuracy unavailable"
    del count
    return rf"held-out acc. ${mean:.2f}\pm{sd:.2f}$"


@dataclass(frozen=True)
class PaperInputs:
    score_bindings: Mapping[tuple[str, int, int], Any]
    causal: Mapping[tuple[str, int, int], Mapping[str, Any]]
    role_results: Mapping[tuple[str, int, int], Mapping[str, Any]]
    counterfactual: Mapping[tuple[str, int, int], Mapping[str, Any]]
    carriage: Mapping[tuple[str, int], Mapping[str, Any]]
    best_seeds: Mapping[tuple[str, int], int]
    performance: Sequence[Mapping[str, Any]]


def _load_causal_artifact(
    *,
    base_canonical_root: Path,
    model: str,
    records: int,
    seed: int,
    checkpoint_sha256: str,
) -> Mapping[str, Any]:
    task = task_name(model, records)
    path = (
        base_canonical_root
        / task
        / f"seed_{int(seed)}"
        / "cache"
        / "causal"
        / "validation.pt"
    )
    artifact = load_cache_artifact_file(path)
    contract = artifact.metadata.get("contract", {})
    actual = (
        str(contract.get("task")),
        int(contract.get("train_seed", -1)),
        str(contract.get("checkpoint_sha256")),
    )
    expected = (task, int(seed), str(checkpoint_sha256))
    if actual != expected:
        raise StaleCacheError(
            f"causal/score provenance mismatch for {task}:seed{seed}: {actual} != {expected}"
        )
    return artifact.value


def _load_carriage_artifact(
    *,
    base_canonical_root: Path,
    model: str,
    records: int,
    seed: int,
    checkpoint_sha256: str,
) -> Mapping[str, Any]:
    task = task_name(model, records)
    path = (
        base_canonical_root
        / task
        / f"seed_{int(seed)}"
        / "cache"
        / "carriage"
        / "fields.pt"
    )
    artifact = load_cache_artifact_file(path)
    contract = artifact.metadata.get("contract", {})
    actual = (
        str(contract.get("task")),
        int(contract.get("train_seed", -1)),
        str(contract.get("checkpoint_sha256")),
    )
    expected = (task, int(seed), str(checkpoint_sha256))
    if actual != expected:
        raise StaleCacheError(
            f"carriage/score provenance mismatch for {task}:seed{seed}: "
            f"{actual} != {expected}"
        )
    return artifact.value


def load_paper_inputs(
    *,
    base_analysis_root: Path,
    base_canonical_root: Path,
    source_extension_root: Path,
    training_run_dir: Path,
    models: Sequence[str],
    score_ns: Sequence[int],
    causal_ns: Sequence[int],
    counterfactual_ns: Sequence[int],
    carriage_ns: Sequence[int],
    cached_ns: Sequence[int],
    performance_ns: Sequence[int],
    seeds: Sequence[int],
    width: int,
    donors_per_role: int,
    accuracy_gate: float,
) -> tuple[PaperInputs, tuple[Any, ...]]:
    """Load and cross-check every protected source artifact used by the paper figures."""

    sizes, numerical, bootstrap, families, analysis_seed = _methodology_policies(
        base_canonical_root
    )
    bindings: dict[tuple[str, int, int], Any] = {}
    for records in score_ns:
        for model in models:
            for seed in seeds:
                bindings[(str(model), int(records), int(seed))] = load_source_binding(
                    base_canonical_root=base_canonical_root,
                    extension_root=source_extension_root,
                    cached_ns=cached_ns,
                    model_name=model,
                    records=records,
                    seed=seed,
                )

    source_store = ExtensionStore(source_extension_root)
    role_results = load_cached_role_results(
        store=source_store,
        base_canonical_root=base_canonical_root,
        extension_root=source_extension_root,
        models=models,
        ns=score_ns,
        cached_ns=cached_ns,
        seeds=seeds,
        numerical=numerical,
        families=families,
        bootstrap=bootstrap,
        analysis_seed=analysis_seed,
    )
    counterfactual_roles = {
        key: value
        for key, value in role_results.items()
        if int(key[1]) in {int(records) for records in counterfactual_ns}
    }
    counterfactual = load_cached_counterfactual_results(
        store=source_store,
        base_canonical_root=base_canonical_root,
        extension_root=source_extension_root,
        models=models,
        ns=counterfactual_ns,
        cached_ns=cached_ns,
        seeds=seeds,
        role_results=counterfactual_roles,
        sizes=sizes,
        numerical=numerical,
        donors_per_role=donors_per_role,
        accuracy_gate=accuracy_gate,
        analysis_seed=analysis_seed,
    )
    performance = _read_performance_rows(
        base_analysis_root=base_analysis_root,
        training_run_dir=training_run_dir,
        models=models,
        ns=performance_ns,
        seeds=seeds,
        width=width,
    )
    best_seeds = best_seed_by_validation(performance)

    causal: dict[tuple[str, int, int], Mapping[str, Any]] = {}
    carriage: dict[tuple[str, int], Mapping[str, Any]] = {}
    missing: list[str] = []
    for records in causal_ns:
        for model in models:
            for seed in seeds:
                binding = bindings.get((str(model), int(records), int(seed)))
                if binding is None:
                    missing.append(f"score:{model}:N{records}:seed{seed}")
                    continue
                try:
                    causal[(str(model), int(records), int(seed))] = _load_causal_artifact(
                        base_canonical_root=base_canonical_root,
                        model=model,
                        records=records,
                        seed=seed,
                        checkpoint_sha256=binding.checkpoint_sha256,
                    )
                except FileNotFoundError:
                    missing.append(f"causal:{model}:N{records}:seed{seed}")
    for records in carriage_ns:
        for model in models:
            key = (str(model), int(records))
            if key not in best_seeds:
                missing.append(f"best-seed:{model}:N{records}")
                continue
            seed = int(best_seeds[key])
            binding = bindings.get((str(model), int(records), seed))
            if binding is None:
                missing.append(f"score:{model}:N{records}:seed{seed}")
                continue
            try:
                carriage[key] = _load_carriage_artifact(
                    base_canonical_root=base_canonical_root,
                    model=model,
                    records=records,
                    seed=seed,
                    checkpoint_sha256=binding.checkpoint_sha256,
                )
            except FileNotFoundError:
                missing.append(f"carriage:{model}:N{records}:seed{seed}")
    if missing:
        raise FileNotFoundError(
            "paper synthesis is missing protected source artifacts:\n- "
            + "\n- ".join(missing)
        )
    return (
        PaperInputs(
            score_bindings=bindings,
            causal=causal,
            role_results=role_results,
            counterfactual=counterfactual,
            carriage=carriage,
            best_seeds=best_seeds,
            performance=performance,
        ),
        (sizes, numerical, bootstrap, families, analysis_seed),
    )


def _family_membership(scores: Mapping[str, Any]) -> dict[tuple[int, int], str]:
    return {
        tuple(map(int, head)): str(name)
        for name, heads in scores.get("families", {}).items()
        for head in heads
    }


def _head_interval_rows(
    inputs: PaperInputs,
    *,
    models: Sequence[str],
    seeds: Sequence[int],
    records: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in models:
        for seed_index, seed in enumerate(seeds):
            binding = inputs.score_bindings[(str(model), int(records), int(seed))]
            scores = binding.score_artifact.value
            coordinates = scores["coordinates"]
            low, high = _interval_arrays(scores["intervals"])
            membership = _family_membership(scores)
            values = (
                np.asarray(coordinates.normalized_structural),
                np.asarray(coordinates.normalized_semantic),
                np.asarray(coordinates.selectivity),
                np.asarray(coordinates.joint_sensitivity),
            )
            lows = (low[3], low[2], low[5], low[4])
            highs = (high[3], high[2], high[5], high[4])
            for layer in range(values[0].shape[0]):
                for head in range(values[0].shape[1]):
                    row = {
                        "model": model,
                        "N": int(records),
                        "seed": int(seed),
                        "seed_marker": SEED_MARKERS[seed_index % len(SEED_MARKERS)],
                        "layer": int(layer),
                        "head": int(head),
                        "family": membership.get((layer, head), ""),
                        "active": bool(coordinates.active[layer, head]),
                    }
                    for name, point, lo, hi in zip(
                        ("s_str", "s_sem", "D_rel", "J"), values, lows, highs
                    ):
                        row[name] = float(point[layer, head])
                        row[f"{name}_low"] = float(lo[layer, head])
                        row[f"{name}_high"] = float(hi[layer, head])
                    rows.append(row)
    return rows


def _family_edge(family: str, active: bool) -> tuple[str, float]:
    if not active:
        return ("white", 0.35)
    if family == "semantic_leaning":
        return (SEMANTIC_COLOUR, 1.15)
    if family == "structural_leaning":
        return (STRUCTURAL_COLOUR, 1.15)
    return ("white", 0.45)


def plot_core_specialisation(
    inputs: PaperInputs,
    *,
    output_dir: Path,
    models: Sequence[str],
    seeds: Sequence[int],
    records: int,
    headline: bool,
) -> None:
    """Mandatory core head landscapes, without whiskers over the scientific scatter."""

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    rows = _head_interval_rows(inputs, models=models, seeds=seeds, records=records)
    figure_dir = (
        output_dir / "figures"
        if headline
        else output_dir / "supplementary" / "core"
    )
    table_name = (
        f"figure_01_core_heads_N{int(records)}.csv"
        if headline
        else f"supplement_core_heads_N{int(records)}.csv"
    )
    _write_csv(output_dir / "tables" / table_name, rows)

    with _style():
        fig, axes = plt.subplots(
            2,
            len(models),
            figsize=(2.75 * len(models), 5.75),
            constrained_layout=False,
            squeeze=False,
        )
        for column, model in enumerate(models):
            model_rows = [row for row in rows if str(row["model"]) == str(model)]
            for row_index, coordinate in enumerate(("scores", "derived")):
                axis = axes[row_index, column]
                if coordinate == "scores":
                    all_values = [
                        value
                        for row in model_rows
                        for value in (float(row["s_str"]), float(row["s_sem"]))
                        if np.isfinite(value)
                    ]
                    upper = max(all_values, default=1.0) * 1.08
                    upper = max(upper, 1.15)
                    axis.plot(
                        [0, upper],
                        [0, upper],
                        color="#8A8A8A",
                        linestyle="--",
                        linewidth=0.75,
                        zorder=0,
                    )
                    axis.set_xlim(-0.03 * upper, upper)
                    axis.set_ylim(-0.03 * upper, upper)
                    axis.set_aspect("equal", adjustable="box")
                    axis.set_xlabel(
                        r"Structural score  $S_{str}/\overline{S}_{str}$"
                    )
                    if column == 0:
                        axis.set_ylabel(
                            r"Semantic score  $S_{sem}/\overline{S}_{sem}$"
                        )
                else:
                    upper = max(
                        [
                            float(row["J"])
                            for row in model_rows
                            if np.isfinite(float(row["J"]))
                        ],
                        default=1.0,
                    ) * 1.08
                    axis.axvline(
                        0,
                        color="#8A8A8A",
                        linestyle="--",
                        linewidth=0.75,
                        zorder=0,
                    )
                    activity_floor = min(
                        [
                            float(row["J"])
                            for row in model_rows
                            if bool(row["active"]) and np.isfinite(float(row["J"]))
                        ],
                        default=float("nan"),
                    )
                    if np.isfinite(activity_floor):
                        axis.axhline(
                            activity_floor,
                            color="#B9B9B9",
                            linestyle=":",
                            linewidth=0.7,
                            zorder=0,
                        )
                    axis.set_xlim(-1.05, 1.05)
                    axis.set_ylim(-0.03 * upper, max(upper, 1.1))
                    axis.set_xlabel(
                        r"Selectivity $D_{rel}$"
                        "\n"
                        r"(structural $\leftarrow 0 \rightarrow$ semantic)"
                    )
                    if column == 0:
                        axis.set_ylabel(r"Joint sensitivity $J$")
                for row in model_rows:
                    active = bool(row["active"])
                    layer = int(row["layer"])
                    edge, linewidth = _family_edge(str(row["family"]), active)
                    face = (
                        LAYER_COLOURS[layer % len(LAYER_COLOURS)]
                        if active
                        else INACTIVE_COLOUR
                    )
                    x = float(row["s_str"] if coordinate == "scores" else row["D_rel"])
                    y = float(row["s_sem"] if coordinate == "scores" else row["J"])
                    axis.scatter(
                        x,
                        y,
                        s=31 if active else 24,
                        marker=str(row["seed_marker"]),
                        facecolor=face,
                        edgecolor=edge,
                        linewidth=linewidth,
                        alpha=0.92 if active else 0.70,
                        zorder=3,
                    )
            axes[0, column].set_title(
                f"{MODEL_LABELS[str(model)]}\n"
                + _accuracy_subtitle(
                    inputs.performance, str(model), int(records), seeds
                ),
                fontsize=10.2,
                pad=8,
            )
        layer_count = max(int(row["layer"]) for row in rows) + 1
        layer_handles = [
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markerfacecolor=LAYER_COLOURS[layer % len(LAYER_COLOURS)],
                markeredgecolor="white",
                label=f"Layer {layer + 1}",
                markersize=5.5,
            )
            for layer in range(layer_count)
        ]
        seed_handles = [
            Line2D(
                [],
                [],
                marker=SEED_MARKERS[index],
                linestyle="none",
                markerfacecolor="#777777",
                markeredgecolor="white",
                label=f"Seed {seed}",
                markersize=5.5,
            )
            for index, seed in enumerate(seeds)
        ]
        family_handles = [
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markerfacecolor="white",
                markeredgecolor=SEMANTIC_COLOUR,
                markeredgewidth=1.2,
                label="Semantic-leaning",
                markersize=5.5,
            ),
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markerfacecolor="white",
                markeredgecolor=STRUCTURAL_COLOUR,
                markeredgewidth=1.2,
                label="Structural-leaning",
                markersize=5.5,
            ),
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markerfacecolor=INACTIVE_COLOUR,
                markeredgecolor="white",
                label="Below activity floor",
                markersize=5.5,
            ),
        ]
        fig.legend(
            handles=[*layer_handles, *seed_handles, *family_handles],
            loc="lower center",
            bbox_to_anchor=(0.5, 0.008),
            ncol=min(8, len(layer_handles) + len(seed_handles) + len(family_handles)),
            fontsize=7.2,
            handletextpad=0.35,
            columnspacing=0.9,
        )
        fig.suptitle(
            rf"Canonical head specialisation at memory size $N={int(records)}$",
            fontsize=13,
            y=0.985,
        )
        fig.subplots_adjust(
            left=0.085,
            right=0.99,
            bottom=0.16,
            top=0.875,
            hspace=0.52,
            wspace=0.30,
        )
        stem = (
            CORE_FIGURE_STEM.format(N=int(records))
            if headline
            else f"S01_core_specialisation_N{int(records)}"
        )
        _save_figure(
            fig,
            figure_dir,
            stem,
            {
                "paper_version": PAPER_VERSION,
                "N": int(records),
                "point": "one attention head; no cross-seed head alignment implied",
                "seed_encoding": "marker shape",
                "layer_encoding": "categorical fill colour",
                "family_encoding": "marker outline",
                "accuracy_subtitle": "mean ± sample SD across three training seeds",
                "scatter_uncertainty": (
                    "moved to model-specific supplementary interval companions"
                ),
            },
        )
        plt.close(fig)

    for model in models:
        plot_head_interval_companion(
            rows,
            output_dir=output_dir,
            model=str(model),
            records=int(records),
        )


def plot_head_interval_companion(
    rows: Sequence[Mapping[str, Any]],
    *,
    output_dir: Path,
    model: str,
    records: int,
) -> None:
    """Point-resolved interval companion required when the main scatter is point-only."""

    import matplotlib.pyplot as plt

    selected = [row for row in rows if str(row["model"]) == str(model)]
    selected = sorted(
        selected,
        key=lambda row: (int(row["seed"]), int(row["layer"]), int(row["head"])),
    )
    if not selected:
        return
    metrics = (
        ("s_str", r"Structural score $s_{str}$"),
        ("s_sem", r"Semantic score $s_{sem}$"),
        ("D_rel", r"Selectivity $D_{rel}$"),
        ("J", r"Joint sensitivity $J$"),
    )
    y = np.arange(len(selected))
    labels = [
        f"S{int(row['seed'])} · L{int(row['layer']) + 1}H{int(row['head']) + 1}"
        for row in selected
    ]
    with _style():
        fig, axes = plt.subplots(
            1,
            4,
            figsize=(8.1, max(5.2, 0.13 * len(selected))),
            sharey=True,
            constrained_layout=False,
        )
        for axis, (metric, label) in zip(axes, metrics):
            for position, row in enumerate(selected):
                point = float(row[metric])
                low = float(row[f"{metric}_low"])
                high = float(row[f"{metric}_high"])
                colour = LAYER_COLOURS[int(row["layer"]) % len(LAYER_COLOURS)]
                axis.plot([low, high], [position, position], color=colour, linewidth=0.65)
                axis.scatter(
                    point,
                    position,
                    marker=str(row["seed_marker"]),
                    s=12,
                    color=colour,
                    edgecolor="white",
                    linewidth=0.25,
                    zorder=2,
                )
            axis.set_xlabel(label)
            if metric == "D_rel":
                axis.axvline(0, color="#888888", linestyle="--", linewidth=0.7)
                axis.set_xlim(-1.05, 1.05)
            axis.invert_yaxis()
        axes[0].set_yticks(y)
        axes[0].set_yticklabels(labels, fontsize=5.8)
        fig.suptitle(
            f"{MODEL_LABELS[model]}: nested 95% head intervals at $N={records}$",
            fontsize=11.5,
            y=0.985,
        )
        fig.subplots_adjust(left=0.14, right=0.99, bottom=0.08, top=0.92, wspace=0.28)
        _save_figure(
            fig,
            output_dir / "supplementary" / "intervals",
            f"head_interval_companion_{model}_N{records}",
            {
                "paper_version": PAPER_VERSION,
                "companion_for": (
                    CORE_FIGURE_STEM.format(N=records)
                    if records
                    else "core_specialisation"
                ),
                "interval": "95% nested percentile interval from canonical score cache",
            },
        )
        plt.close(fig)


def conditional_source_scores(scores: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Four input-defined score strata, all using unconditional core-channel means."""

    result: dict[str, np.ndarray] = {}
    for channel in ("semantic", "structural"):
        channel_value = scores["channels"][channel]
        events = list(channel_value.get("events", ()))
        if not events:
            raise ValueError(f"{channel} score cache has no retained event rows")
        raw = np.asarray(channel_value["raw"], dtype=np.float64)
        normalizer = float(np.mean(raw))
        if not np.isfinite(normalizer) or normalizer <= 0:
            raise ValueError(f"{channel} unconditional score mean is not estimable")
        for role, predicate in (
            ("query", lambda source: int(source) == 2),
            ("record", lambda source: int(source) >= 3),
        ):
            selected = [row for row in events if predicate(row["source"])]
            if not selected:
                raise ValueError(f"{channel} cache has no {role} source events")
            estimate, _, _ = aggregate_event_scores(
                np.stack([np.asarray(row["score"], dtype=np.float64) for row in selected]),
                [int(row["graph_id"]) for row in selected],
                [int(row["source"]) for row in selected],
            )
            result[f"{channel}_{role}"] = np.asarray(estimate) / normalizer
        reconstructed = 0.5 * (
            result[f"{channel}_query"] + result[f"{channel}_record"]
        )
        expected = raw / normalizer
        result[f"{channel}_reconstruction_error"] = np.asarray(
            np.max(np.abs(reconstructed - expected))
        )
    return result


def conditional_family_rows(
    inputs: PaperInputs,
    *,
    models: Sequence[str],
    seeds: Sequence[int],
    records: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in models:
        for seed in seeds:
            binding = inputs.score_bindings[(str(model), int(records), int(seed))]
            scores = binding.score_artifact.value
            conditional = conditional_source_scores(scores)
            for family_name in CORE_FAMILIES:
                family = tuple(tuple(map(int, head)) for head in scores["families"][family_name])
                if not family:
                    continue
                for component in CONDITIONAL_COMPONENTS:
                    values = np.asarray(
                        [conditional[component][head] for head in family],
                        dtype=np.float64,
                    )
                    rows.append(
                        {
                            "model": model,
                            "N": int(records),
                            "seed": int(seed),
                            "family": family_name,
                            "component": component,
                            "estimate": float(np.mean(values)),
                            "heads": len(family),
                            "semantic_reconstruction_error": float(
                                conditional["semantic_reconstruction_error"]
                            ),
                            "structural_reconstruction_error": float(
                                conditional["structural_reconstruction_error"]
                            ),
                            "normalisation": (
                                "unconditional discovery-split core channel mean"
                            ),
                        }
                    )
    return rows


def _association_record(
    causal: Mapping[str, Any],
    key: str,
    *,
    active: bool,
) -> tuple[float, float, int]:
    record = causal.get("associations", {}).get(key, {})
    pooled_key = "pooled_active" if active else "pooled"
    pooled = record.get(pooled_key, {})
    permutation_key = (
        "within_layer_permutation_active"
        if active
        else "within_layer_permutation"
    )
    permutation = record.get(permutation_key, {})
    return (
        _as_float(pooled.get("rho")),
        _as_float(permutation.get("p")),
        int(pooled.get("n", record.get("n_active", 0)) or 0),
    )


def _family_channel_interaction(
    causal: Mapping[str, Any],
    *,
    endpoint: str,
    control: bool,
) -> float:
    targets = causal.get("summary", {}).get("targets", {})
    prefix = "g" if endpoint == "gross" else "n"
    if control:
        sem_name = "control_semantic_leaning_central_control"
        str_name = "control_structural_leaning_central_control"
    else:
        sem_name = "family_semantic_leaning"
        str_name = "family_structural_leaning"
    if sem_name not in targets or str_name not in targets:
        return float("nan")
    sem = targets[sem_name]["calibrated"]
    struct = targets[str_name]["calibrated"]
    sem_contrast = _as_float(sem.get(f"{prefix}_semantic")) - _as_float(
        sem.get(f"{prefix}_structural")
    )
    str_contrast = _as_float(struct.get(f"{prefix}_semantic")) - _as_float(
        struct.get(f"{prefix}_structural")
    )
    return float(sem_contrast - str_contrast)


def causal_summary_rows(
    inputs: PaperInputs,
    *,
    models: Sequence[str],
    causal_ns: Sequence[int],
    seeds: Sequence[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    raw_tests = (
        (
            "S_semantic_vs_G_semantic",
            r"$S_{\rm sem}\leftrightarrow G_{\rm sem}$",
            "same_channel",
        ),
        (
            "S_structural_vs_G_structural",
            r"$S_{\rm str}\leftrightarrow G_{\rm str}$",
            "same_channel",
        ),
        (
            "S_semantic_vs_G_structural_control",
            r"$S_{\rm sem}\leftrightarrow G_{\rm str}$",
            "cross_channel",
        ),
        (
            "S_structural_vs_G_semantic_control",
            r"$S_{\rm str}\leftrightarrow G_{\rm sem}$",
            "cross_channel",
        ),
    )
    coordinate_tests = (
        ("J_vs_clean_prediction_movement", r"$J\leftrightarrow$ clean ablation", False),
        ("J_vs_gross_total", r"$J\leftrightarrow$ total gross", False),
        ("J_vs_necessity_total", r"$J\leftrightarrow$ total necessity", False),
        (
            "D_rel_vs_gross_contrast",
            r"$D_{\rm rel}\leftrightarrow$ gross contrast",
            True,
        ),
        (
            "D_rel_vs_necessity_contrast",
            r"$D_{\rm rel}\leftrightarrow$ necessity contrast",
            True,
        ),
    )
    for records in causal_ns:
        for model in models:
            for seed in seeds:
                causal = inputs.causal.get((str(model), int(records), int(seed)))
                if causal is None:
                    continue
                raw = causal.get("associations", {}).get("raw_score_validation", {})
                for key, label, kind in raw_tests:
                    record = raw.get(key, {})
                    rows.append(
                        {
                            "model": model,
                            "N": int(records),
                            "seed": int(seed),
                            "section": "raw_score_calibration",
                            "test": key,
                            "label": label,
                            "kind": kind,
                            "estimate": _as_float(record.get("rho")),
                            "p_within_layer": float("nan"),
                            "n": int(record.get("n", 0) or 0),
                        }
                    )
                for key, label, active in coordinate_tests:
                    rho, p_value, count = _association_record(
                        causal, key, active=active
                    )
                    rows.append(
                        {
                            "model": model,
                            "N": int(records),
                            "seed": int(seed),
                            "section": "coordinate_validation",
                            "test": key,
                            "label": label,
                            "kind": "D_rel" if active else "J",
                            "estimate": rho,
                            "p_within_layer": p_value,
                            "n": count,
                        }
                    )
                for endpoint in ("gross", "necessity"):
                    for control in (False, True):
                        rows.append(
                            {
                                "model": model,
                                "N": int(records),
                                "seed": int(seed),
                                "section": "family_interaction",
                                "test": f"{endpoint}_{'control' if control else 'family'}",
                                "label": endpoint.capitalize(),
                                "kind": "matched_control" if control else "core_families",
                                "estimate": _family_channel_interaction(
                                    causal,
                                    endpoint=endpoint,
                                    control=control,
                                ),
                                "p_within_layer": float("nan"),
                                "n": 2,
                            }
                        )
                clean = causal.get("clean_ablation", {})
                for target in (
                    "family_semantic_leaning",
                    "family_structural_leaning",
                    "control_semantic_leaning_central_control",
                    "control_structural_leaning_central_control",
                ):
                    if target not in clean:
                        continue
                    rows.append(
                        {
                            "model": model,
                            "N": int(records),
                            "seed": int(seed),
                            "section": "family_clean_ablation",
                            "test": target,
                            "label": target,
                            "kind": "family_endpoint",
                            "estimate": _as_float(
                                clean[target].get("prediction_movement")
                            ),
                            "loss_change": _as_float(
                                clean[target].get("loss_change")
                            ),
                            "p_within_layer": float("nan"),
                            "n": 1,
                        }
                    )
    return rows


def _scatter_seed_summary(
    axis: Any,
    rows: Sequence[Mapping[str, Any]],
    labels: Sequence[str],
    *,
    seeds: Sequence[int],
    colour_by_label: Mapping[str, str],
) -> None:
    for position, label in enumerate(labels):
        chosen = [row for row in rows if str(row["label"]) == str(label)]
        values = []
        for row in chosen:
            value = _as_float(row["estimate"])
            if not np.isfinite(value):
                continue
            seed = int(row["seed"])
            seed_position = list(seeds).index(seed) if seed in seeds else 0
            axis.scatter(
                value,
                position,
                marker=SEED_MARKERS[seed_position % len(SEED_MARKERS)],
                s=28,
                color=colour_by_label.get(label, "#666666"),
                edgecolor="white",
                linewidth=0.45,
                alpha=0.88,
                zorder=3,
            )
            values.append(value)
        if values:
            axis.scatter(
                float(np.mean(values)),
                position,
                marker="D",
                s=17,
                facecolor="white",
                edgecolor="#202020",
                linewidth=0.75,
                zorder=4,
            )
    axis.set_yticks(range(len(labels)))
    axis.set_yticklabels(labels)
    axis.invert_yaxis()
    axis.axvline(0, color="#888888", linestyle="--", linewidth=0.7, zorder=0)


def plot_core_causal_validation(
    inputs: PaperInputs,
    *,
    output_dir: Path,
    models: Sequence[str],
    seeds: Sequence[int],
    records: int,
    all_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Cross-seed synthesis of the README-mandated causal validation programme."""

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    raw_labels = (
        r"$S_{\rm sem}\leftrightarrow G_{\rm sem}$",
        r"$S_{\rm str}\leftrightarrow G_{\rm str}$",
        r"$S_{\rm sem}\leftrightarrow G_{\rm str}$",
        r"$S_{\rm str}\leftrightarrow G_{\rm sem}$",
    )
    coordinate_labels = (
        r"$J\leftrightarrow$ clean ablation",
        r"$J\leftrightarrow$ total gross",
        r"$J\leftrightarrow$ total necessity",
        r"$D_{\rm rel}\leftrightarrow$ gross contrast",
        r"$D_{\rm rel}\leftrightarrow$ necessity contrast",
    )
    with _style():
        fig, axes = plt.subplots(
            3,
            len(models),
            figsize=(3.05 * len(models), 7.0),
            constrained_layout=False,
            squeeze=False,
            gridspec_kw={"height_ratios": (1.0, 1.15, 0.92)},
        )
        for column, model in enumerate(models):
            selected = [
                row
                for row in all_rows
                if str(row["model"]) == str(model) and int(row["N"]) == int(records)
            ]
            raw = [row for row in selected if row["section"] == "raw_score_calibration"]
            _scatter_seed_summary(
                axes[0, column],
                raw,
                raw_labels,
                seeds=seeds,
                colour_by_label={
                    raw_labels[0]: SEMANTIC_COLOUR,
                    raw_labels[1]: STRUCTURAL_COLOUR,
                    raw_labels[2]: CONTROL_COLOUR,
                    raw_labels[3]: CONTROL_COLOUR,
                },
            )
            axes[0, column].set_xlim(-1.05, 1.05)
            axes[0, column].set_xlabel(r"Spearman $\rho$")
            axes[0, column].set_title(
                f"{MODEL_LABELS[str(model)]}\n"
                + _accuracy_subtitle(
                    inputs.performance, str(model), int(records), seeds
                ),
                fontsize=10.2,
                pad=8,
            )
            coordinate = [
                row for row in selected if row["section"] == "coordinate_validation"
            ]
            _scatter_seed_summary(
                axes[1, column],
                coordinate,
                coordinate_labels,
                seeds=seeds,
                colour_by_label={
                    label: ("#8A5A2B" if label.startswith("J") else "#76528B")
                    for label in coordinate_labels
                },
            )
            axes[1, column].set_xlim(-1.05, 1.05)
            axes[1, column].set_xlabel(r"Spearman $\rho$")
            if column > 0:
                for row_index in (0, 1):
                    axes[row_index, column].set_yticklabels([])
                    axes[row_index, column].tick_params(axis="y", length=0)
            interaction = [
                row for row in selected if row["section"] == "family_interaction"
            ]
            positions = {"Gross": 0, "Necessity": 1}
            for label, position in positions.items():
                for kind, offset, colour, marker in (
                    ("core_families", -0.10, "#2D5F73", "o"),
                    ("matched_control", 0.10, CONTROL_COLOUR, "s"),
                ):
                    chosen = [
                        row
                        for row in interaction
                        if row["label"] == label and row["kind"] == kind
                    ]
                    values = []
                    for row in chosen:
                        value = _as_float(row["estimate"])
                        if not np.isfinite(value):
                            continue
                        seed_position = list(seeds).index(int(row["seed"]))
                        axes[2, column].scatter(
                            position + offset,
                            value,
                            marker=SEED_MARKERS[seed_position],
                            s=26,
                            color=colour,
                            edgecolor="white",
                            linewidth=0.4,
                            zorder=3,
                        )
                        values.append(value)
                    if values:
                        axes[2, column].scatter(
                            position + offset,
                            float(np.mean(values)),
                            marker=marker,
                            s=18,
                            facecolor="white",
                            edgecolor="#222222",
                            linewidth=0.7,
                            zorder=4,
                        )
            axes[2, column].axhline(
                0, color="#888888", linestyle="--", linewidth=0.7
            )
            axes[2, column].set_xticks((0, 1))
            axes[2, column].set_xticklabels(("Gross patch", "Necessity"))
            axes[2, column].set_ylabel(
                "Family × channel\ncausal interaction"
                if column == 0
                else ""
            )
        axes[0, 0].set_ylabel("Raw-score\ncalibration")
        axes[1, 0].set_ylabel("Coordinate\nvalidation")
        seed_handles = [
            Line2D(
                [],
                [],
                marker=SEED_MARKERS[index],
                linestyle="none",
                markerfacecolor="#666666",
                markeredgecolor="white",
                label=f"Seed {seed}",
                markersize=5.5,
            )
            for index, seed in enumerate(seeds)
        ]
        semantic_handle = Line2D(
            [], [], color=SEMANTIC_COLOUR, marker="o", linestyle="none", label="Semantic"
        )
        structural_handle = Line2D(
            [], [], color=STRUCTURAL_COLOUR, marker="o", linestyle="none", label="Structural"
        )
        control_handle = Line2D(
            [], [], color=CONTROL_COLOUR, marker="s", linestyle="none", label="Matched control"
        )
        fig.legend(
            handles=[*seed_handles, semantic_handle, structural_handle, control_handle],
            loc="lower center",
            bbox_to_anchor=(0.5, 0.008),
            ncol=6,
            fontsize=7.4,
        )
        fig.suptitle(
            rf"Canonical scores predict held-out causal function at $N={records}$",
            fontsize=13,
            y=0.988,
        )
        fig.subplots_adjust(
            left=0.13,
            right=0.99,
            bottom=0.10,
            top=0.89,
            hspace=0.54,
            wspace=0.24,
        )
        _save_figure(
            fig,
            output_dir / "figures",
            CAUSAL_FIGURE_STEM.format(N=int(records)),
            {
                "paper_version": PAPER_VERSION,
                "N": int(records),
                "raw_score_controls": "cross-channel relationships",
                "coordinate_tests": (
                    "J against unsigned importance/total channel response; "
                    "D_rel against signed channel contrasts"
                ),
                "family_interaction": (
                    "[semantic-structural response of semantic family] - "
                    "[semantic-structural response of structural family]"
                ),
                "replication": "seed points plus an unfilled descriptive mean diamond",
            },
        )
        plt.close(fig)


def _counterfactual_vector_transform(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64).reshape(8, 2)
    ratios = value[:, 0] / value[:, 1]
    family = (ratios[0] - ratios[1]) - (ratios[2] - ratios[3])
    control = (ratios[4] - ratios[5]) - (ratios[6] - ratios[7])
    return np.asarray((family, control, family - control), dtype=np.float64)


def counterfactual_graph_observations(
    results: Mapping[tuple[str, int, int], Mapping[str, Any]],
    *,
    model: str,
    records: int,
    require_correct: bool,
) -> tuple[list[Observation], dict[str, int]]:
    """Graph-level numerator/denominator vectors for a paired double dissociation."""

    cells = (
        ("query", "address"),
        ("query", "content"),
        ("value", "address"),
        ("value", "content"),
        ("query", "address_control"),
        ("query", "content_control"),
        ("value", "address_control"),
        ("value", "content_control"),
    )
    observations: list[Observation] = []
    eligible_events = 0
    correct_events = 0
    seeds_used: set[int] = set()
    for (cell_model, cell_n, seed), result in results.items():
        if str(cell_model) != str(model) or int(cell_n) != int(records):
            continue
        if not bool(result.get("primary")):
            continue
        rows = list(result.get("rows", ()))
        for row in rows:
            if bool(row.get("estimable")):
                eligible_events += 1
                if bool(row.get("clean_correct")) and bool(
                    row.get("counterfactual_correct")
                ):
                    correct_events += 1
        graph_ids = sorted({int(row["graph_id"]) for row in rows})
        for graph in graph_ids:
            vector: list[float] = []
            complete = True
            for role, family in cells:
                selected = [
                    row
                    for row in rows
                    if int(row["graph_id"]) == graph
                    and str(row["role"]) == role
                    and str(row["family"]) == family
                    and bool(row.get("estimable"))
                    and (
                        not require_correct
                        or (
                            bool(row.get("clean_correct"))
                            and bool(row.get("counterfactual_correct"))
                        )
                    )
                    and np.isfinite(_as_float(row.get("directional_numerator")))
                    and np.isfinite(_as_float(row.get("full_margin_change")))
                ]
                if not selected:
                    complete = False
                    break
                numerator = float(
                    np.mean(
                        [_as_float(row["directional_numerator"]) for row in selected]
                    )
                )
                denominator = float(
                    np.mean([_as_float(row["full_margin_change"]) for row in selected])
                )
                if not np.isfinite(denominator) or denominator <= 0:
                    complete = False
                    break
                vector.extend((numerator, denominator))
            if complete:
                seeds_used.add(int(seed))
                observations.append(
                    Observation(
                        seed=int(seed),
                        graph=int(graph),
                        source=0,
                        donor=0,
                        value=np.asarray(vector, dtype=np.float64),
                    )
                )
    return observations, {
        "graphs": len(observations),
        "seeds": len(seeds_used),
        "eligible_event_rows": int(eligible_events),
        "correct_event_rows": int(correct_events),
    }


def counterfactual_double_dissociation_rows(
    results: Mapping[tuple[str, int, int], Mapping[str, Any]],
    *,
    models: Sequence[str],
    ns: Sequence[int],
    bootstrap: BootstrapPolicy,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    for model in models:
        for records in ns:
            for require_correct, analysis_set in (
                (True, "successful_exact_recall"),
                (False, "all_estimable_events"),
            ):
                observations, support = counterfactual_graph_observations(
                    results,
                    model=str(model),
                    records=int(records),
                    require_correct=require_correct,
                )
                if not observations:
                    continue
                offset += 1
                policy = dataclasses.replace(
                    bootstrap,
                    rng_seed=int(bootstrap.rng_seed) + 70_000 + offset,
                    resample_source=False,
                    resample_donor=False,
                )
                interval = nested_percentile_interval(
                    observations,
                    policy,
                    transform=_counterfactual_vector_transform,
                )
                for index, estimand in enumerate(
                    ("family_double_dissociation", "control_double_dissociation", "adjusted")
                ):
                    rows.append(
                        {
                            "model": model,
                            "N": int(records),
                            "analysis_set": analysis_set,
                            "estimand": estimand,
                            "estimate": float(interval.estimate[index]),
                            "ci95_low": float(interval.low[index]),
                            "ci95_high": float(interval.high[index]),
                            **support,
                            "estimator": (
                                "paired graph-balanced ratio-of-means; "
                                "seed and graph bootstrap"
                            ),
                        }
                    )
    return rows


def plot_role_and_counterfactual(
    inputs: PaperInputs,
    *,
    output_dir: Path,
    models: Sequence[str],
    seeds: Sequence[int],
    records: int,
    counterfactual_ns: Sequence[int],
    conditional_rows: Sequence[Mapping[str, Any]],
    counterfactual_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Core-family source-role fingerprints plus exact paired counterfactual validation."""

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    family_colours = {
        "semantic_leaning": SEMANTIC_COLOUR,
        "structural_leaning": STRUCTURAL_COLOUR,
    }
    primary_counterfactual = [
        row
        for row in counterfactual_rows
        if row["analysis_set"] == "successful_exact_recall"
        and row["estimand"] == "adjusted"
    ]
    finite_limits = np.asarray(
        [
            float(row[key])
            for row in primary_counterfactual
            for key in ("ci95_low", "ci95_high")
            if np.isfinite(_as_float(row.get(key)))
        ],
        dtype=np.float64,
    )
    if finite_limits.size:
        lower = min(float(np.min(finite_limits)), 0.0)
        upper = max(float(np.max(finite_limits)), 0.0)
        padding = max(0.08 * (upper - lower), 0.03)
        forest_limits = (lower - padding, upper + padding)
    else:
        forest_limits = (-1.0, 1.0)
    with _style():
        fig, axes = plt.subplots(
            2,
            len(models),
            figsize=(3.1 * len(models), 5.75),
            constrained_layout=False,
            squeeze=False,
            gridspec_kw={"height_ratios": (1.12, 0.88)},
        )
        x = np.arange(len(CONDITIONAL_COMPONENTS))
        for column, model in enumerate(models):
            selected = [
                row
                for row in conditional_rows
                if str(row["model"]) == str(model) and int(row["N"]) == int(records)
            ]
            for family in CORE_FAMILIES:
                for seed_index, seed in enumerate(seeds):
                    values = []
                    for component in CONDITIONAL_COMPONENTS:
                        match = [
                            row
                            for row in selected
                            if row["family"] == family
                            and int(row["seed"]) == int(seed)
                            and row["component"] == component
                        ]
                        values.append(
                            _as_float(match[0]["estimate"]) if match else np.nan
                        )
                    axes[0, column].plot(
                        x,
                        values,
                        color=family_colours[family],
                        linewidth=0.75,
                        alpha=0.35,
                    )
                    axes[0, column].scatter(
                        x,
                        values,
                        marker=SEED_MARKERS[seed_index],
                        s=27,
                        color=family_colours[family],
                        edgecolor="white",
                        linewidth=0.4,
                        zorder=3,
                    )
            axes[0, column].axvline(
                1.5, color="#A0A0A0", linestyle="--", linewidth=0.7
            )
            axes[0, column].set_xticks(x)
            axes[0, column].set_xticklabels(CONDITIONAL_LABELS, fontsize=7.1)
            axes[0, column].set_ylabel(
                "Conditional sensitivity\n"
                "(unconditional channel scale)"
                if column == 0
                else ""
            )
            axes[0, column].set_title(
                f"{MODEL_LABELS[str(model)]}\n"
                + _accuracy_subtitle(
                    inputs.performance, str(model), int(records), seeds
                ),
                fontsize=10.2,
                pad=8,
            )

            forest = [
                row
                for row in counterfactual_rows
                if str(row["model"]) == str(model)
                and row["analysis_set"] == "successful_exact_recall"
                and row["estimand"] == "adjusted"
            ]
            forest = sorted(forest, key=lambda row: int(row["N"]))
            y = np.arange(len(forest))
            for position, row in enumerate(forest):
                estimate = float(row["estimate"])
                low = float(row["ci95_low"])
                high = float(row["ci95_high"])
                axes[1, column].plot(
                    [low, high],
                    [position, position],
                    color=MODEL_COLOURS[str(model)],
                    linewidth=1.15,
                    zorder=2,
                )
                axes[1, column].scatter(
                    estimate,
                    position,
                    marker=MODEL_MARKERS[str(model)],
                    s=34,
                    color=MODEL_COLOURS[str(model)],
                    edgecolor="white",
                    linewidth=0.5,
                    zorder=3,
                )
            axes[1, column].axvline(
                0, color="#888888", linestyle="--", linewidth=0.75
            )
            axes[1, column].set_xlim(*forest_limits)
            axes[1, column].set_yticks(y)
            compact_labels = []
            for row in forest:
                accuracy_mean, accuracy_sd, _ = accuracy_summary(
                    inputs.performance,
                    str(model),
                    int(row["N"]),
                    seeds,
                )
                if np.isfinite(accuracy_mean):
                    compact_labels.append(
                        f"$N={int(row['N'])}$\n"
                        rf"acc. ${accuracy_mean:.2f}\pm{accuracy_sd:.2f}$"
                    )
                else:
                    compact_labels.append(f"$N={int(row['N'])}$\nacc. unavailable")
            axes[1, column].set_yticklabels(
                compact_labels,
                fontsize=6.4,
                linespacing=1.05,
            )
            axes[1, column].tick_params(axis="x", labelsize=7.0)
            if forest:
                axes[1, column].set_ylim(-0.55, len(forest) - 0.45)
            if not forest:
                axes[1, column].text(
                    0.5,
                    0.5,
                    "No cell passed the\nregistered accuracy/correctness gates",
                    transform=axes[1, column].transAxes,
                    ha="center",
                    va="center",
                    fontsize=8,
                )
                axes[1, column].set_yticks([])
        handles = [
            Line2D(
                [],
                [],
                color=SEMANTIC_COLOUR,
                marker="o",
                label="Canonical semantic-leaning family",
            ),
            Line2D(
                [],
                [],
                color=STRUCTURAL_COLOUR,
                marker="o",
                label="Canonical structural-leaning family",
            ),
            *[
                Line2D(
                    [],
                    [],
                    color="#666666",
                    marker=SEED_MARKERS[index],
                    linestyle="none",
                    label=f"Seed {seed}",
                )
                for index, seed in enumerate(seeds)
            ],
        ]
        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.012),
            ncol=5,
            fontsize=7.2,
        )
        fig.text(
            0.55,
            0.096,
            "Control-adjusted counterfactual double dissociation",
            ha="center",
            va="center",
            fontsize=8.8,
        )
        fig.suptitle(
            "Task-defined roles interpret core specialisation and test semantic subroles",
            fontsize=13,
            y=0.982,
        )
        fig.text(
            0.012,
            0.69,
            "A",
            fontsize=11,
            fontweight="bold",
        )
        fig.text(
            0.012,
            0.365,
            "B",
            fontsize=11,
            fontweight="bold",
        )
        fig.subplots_adjust(
            left=0.105,
            right=0.99,
            bottom=0.18,
            top=0.865,
            hspace=0.36,
            wspace=0.38,
        )
        _save_figure(
            fig,
            output_dir / "figures",
            INTERPRETATION_FIGURE_STEM,
            {
                "paper_version": PAPER_VERSION,
                "fingerprint_N": int(records),
                "conditional_normalisation": (
                    "unconditional core channel mean, never role-specific renormalisation"
                ),
                "counterfactual_primary_set": (
                    "accuracy-eligible checkpoints and events on which both clean and complete "
                    "counterfactual models return their exact respective answers"
                ),
                "counterfactual_estimand": (
                    "family double dissociation minus separately matched-control double "
                    "dissociation; graph-paired ratio-of-means"
                ),
                "counterfactual_family_scope": (
                    "address/content families independently frozen by the v1 semantic-subrole "
                    "extension; this panel interprets semantic subroles and is not a substitute "
                    "for the canonical semantic/structural family interaction in Figure 02"
                ),
                "counterfactual_N_values": list(map(int, counterfactual_ns)),
            },
        )
        plt.close(fig)


def _metric_matrix(
    rows: Sequence[Mapping[str, Any]],
    *,
    models: Sequence[str],
    ns: Sequence[int],
    section: str,
    test: str,
) -> tuple[np.ndarray, np.ndarray]:
    means = np.full((len(models), len(ns)), np.nan, dtype=np.float64)
    sds = np.full_like(means, np.nan)
    for model_index, model in enumerate(models):
        for n_index, records in enumerate(ns):
            values = np.asarray(
                [
                    _as_float(row["estimate"])
                    for row in rows
                    if str(row["model"]) == str(model)
                    and int(row["N"]) == int(records)
                    and row["section"] == section
                    and row["test"] == test
                ],
                dtype=np.float64,
            )
            values = values[np.isfinite(values)]
            if len(values):
                means[model_index, n_index] = float(np.mean(values))
                sds[model_index, n_index] = (
                    float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                )
    return means, sds


def _annotated_matrix(
    fig: Any,
    axis: Any,
    mean: np.ndarray,
    sd: np.ndarray,
    *,
    models: Sequence[str],
    ns: Sequence[int],
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
) -> None:
    image = axis.imshow(mean, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    for row in range(mean.shape[0]):
        for column in range(mean.shape[1]):
            if not np.isfinite(mean[row, column]):
                text = "—"
            else:
                text = f"{mean[row, column]:.2f}\n±{sd[row, column]:.2f}"
            colour = "white" if np.isfinite(mean[row, column]) and abs(mean[row, column]) > (
                0.55 * max(abs(vmin), abs(vmax))
            ) else "#202020"
            axis.text(
                column,
                row,
                text,
                ha="center",
                va="center",
                fontsize=6.5,
                color=colour,
            )
    axis.set_xticks(range(len(ns)))
    axis.set_xticklabels([str(value) for value in ns])
    axis.set_yticks(range(len(models)))
    axis.set_yticklabels([MODEL_LABELS[str(model)] for model in models], fontsize=7.4)
    axis.set_xlabel("Memory size, $N$")
    axis.set_title(title, fontsize=9.4)
    colourbar = fig.colorbar(image, ax=axis, fraction=0.055, pad=0.025)
    colourbar.ax.tick_params(labelsize=6.5)


def plot_competence_and_causal_grounding(
    inputs: PaperInputs,
    *,
    output_dir: Path,
    models: Sequence[str],
    performance_ns: Sequence[int],
    causal_ns: Sequence[int],
    seeds: Sequence[int],
    causal_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Capacity comparison using core causal validation rather than R_role."""

    import matplotlib.pyplot as plt

    j_mean, j_sd = _metric_matrix(
        causal_rows,
        models=models,
        ns=causal_ns,
        section="coordinate_validation",
        test="J_vs_clean_prediction_movement",
    )
    d_mean, d_sd = _metric_matrix(
        causal_rows,
        models=models,
        ns=causal_ns,
        section="coordinate_validation",
        test="D_rel_vs_gross_contrast",
    )
    family_mean, family_sd = _metric_matrix(
        causal_rows,
        models=models,
        ns=causal_ns,
        section="family_interaction",
        test="gross_family",
    )
    with _style():
        fig, axes = plt.subplots(
            1,
            4,
            figsize=(10.8, 3.45),
            constrained_layout=False,
            gridspec_kw={"width_ratios": (1.6, 1, 1, 1)},
        )
        for model in models:
            means, sds = [], []
            for records in performance_ns:
                mean, sd, _ = accuracy_summary(
                    inputs.performance, str(model), int(records), seeds
                )
                means.append(mean)
                sds.append(sd)
            means_array = np.asarray(means)
            sds_array = np.asarray(sds)
            axes[0].plot(
                performance_ns,
                means_array,
                color=MODEL_COLOURS[str(model)],
                marker=MODEL_MARKERS[str(model)],
                linewidth=1.45,
                markersize=4.5,
                label=MODEL_LABELS[str(model)],
            )
            axes[0].fill_between(
                performance_ns,
                np.clip(means_array - sds_array, 0, 1),
                np.clip(means_array + sds_array, 0, 1),
                color=MODEL_COLOURS[str(model)],
                alpha=0.12,
                linewidth=0,
            )
        axes[0].set_xticks(performance_ns)
        axes[0].set_ylim(-0.02, 1.02)
        axes[0].set_xlabel("Memory size, $N$")
        axes[0].set_ylabel("Held-out accuracy")
        axes[0].set_title("A  Retrieval competence", fontsize=9.8)
        handles, labels = axes[0].get_legend_handles_labels()
        _annotated_matrix(
            fig,
            axes[1],
            j_mean,
            j_sd,
            models=models,
            ns=causal_ns,
            title=r"B  $J$ ↔ clean ablation",
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
        )
        _annotated_matrix(
            fig,
            axes[2],
            d_mean,
            d_sd,
            models=models,
            ns=causal_ns,
            title=r"C  $D_{rel}$ ↔ channel contrast",
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
        )
        finite_family = np.abs(family_mean[np.isfinite(family_mean)])
        sorted_family = np.sort(finite_family)
        largest = float(sorted_family[-1]) if sorted_family.size else 1.0
        second_largest = (
            float(sorted_family[-2]) if sorted_family.size > 1 else largest
        )
        interaction_display_clipped = bool(
            largest > 5.0 * max(second_largest, 0.25)
        )
        limit = max(
            second_largest if interaction_display_clipped else largest,
            0.25,
        )
        _annotated_matrix(
            fig,
            axes[3],
            family_mean,
            family_sd,
            models=models,
            ns=causal_ns,
            title=(
                "D  Family × channel interaction$^{\\dagger}$"
                if interaction_display_clipped
                else "D  Family × channel interaction"
            ),
            cmap="PuOr_r",
            vmin=-limit,
            vmax=limit,
        )
        if interaction_display_clipped:
            fig.text(
                0.992,
                0.145,
                r"$^{\dagger}$Colour scale clipped; annotated estimates are exact.",
                ha="right",
                va="bottom",
                fontsize=6.2,
                color="#555555",
            )
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.008),
            ncol=len(handles),
            fontsize=7.2,
        )
        fig.suptitle(
            "Retrieval competence and causal grounding across capacity regimes",
            fontsize=13,
            y=0.985,
        )
        fig.subplots_adjust(
            left=0.065,
            right=0.995,
            bottom=0.24,
            top=0.82,
            wspace=0.52,
        )
        _save_figure(
            fig,
            output_dir / "figures",
            TRANSITION_FIGURE_STEM,
            {
                "paper_version": PAPER_VERSION,
                "accuracy": "mean with ±1 sample-SD band across training seeds",
                "matrix_cells": "mean ± sample SD across training seeds",
                "causal_N_values": list(map(int, causal_ns)),
                "interpretation": (
                    "tests whether visually observed specialisation remains causally grounded "
                    "as task competence changes"
                ),
                "excluded_metric": (
                    "R_role is intentionally absent: it is amplitude-free and is not a core "
                    "methodology coordinate"
                ),
                "family_interaction_colour_scale": (
                    "clipped at the second-largest absolute cell to keep non-outlier cells "
                    "visible; annotations always show the unclipped mean and SD"
                    if interaction_display_clipped
                    else "full observed range"
                ),
            },
        )
        plt.close(fig)


def role_conditioned_carriage_profiles(
    carriage: Mapping[str, Any],
    *,
    seed: int,
    bootstrap: BootstrapPolicy,
    max_points: int = 14,
    distance_axis: Any | None = None,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Primary raw Functional carriage by source role and canonical channel."""

    output: dict[tuple[str, str], dict[str, Any]] = {}
    offset = 0
    for channel in ("semantic", "structural"):
        rows = list(carriage.get("channels", {}).get(channel, {}).get("pairs", ()))
        if not rows:
            continue
        finite = [
            int(float(row["distance"]))
            for row in rows
            if np.isfinite(_as_float(row.get("distance")))
        ]
        if not finite:
            continue
        axis = (
            distance_axis
            if distance_axis is not None
            else display_bins(tuple(range(max(finite) + 1)), max_points=max_points)
        )
        for role, predicate in (
            ("query", lambda source: int(source) == 2),
            ("record", lambda source: int(source) >= 3),
        ):
            role_rows = [row for row in rows if predicate(row["source"])]
            estimates: list[float] = []
            lows: list[float] = []
            highs: list[float] = []
            graph_support: list[int] = []
            pair_support: list[int] = []
            event_support: list[int] = []
            for group in axis.groups:
                distances = {int(value) for value in group}
                selected = [
                    row
                    for row in role_rows
                    if np.isfinite(_as_float(row.get("distance")))
                    and int(float(row["distance"])) in distances
                    and np.isfinite(_as_float(row.get("F_sens")))
                ]
                graphs = {int(row["graph_id"]) for row in selected}
                pairs = {
                    (
                        int(row["graph_id"]),
                        int(row["carrier"]),
                        int(row["source"]),
                    )
                    for row in selected
                }
                grouped: dict[tuple[int, int, int], list[float]] = {}
                for row in selected:
                    key = (
                        int(row["graph_id"]),
                        int(row["source"]),
                        int(row["donor"]),
                    )
                    grouped.setdefault(key, []).append(float(row["F_sens"]))
                graph_support.append(len(graphs))
                pair_support.append(len(pairs))
                event_support.append(len(grouped))
                if (
                    len(graphs) < int(bootstrap.minimum_graphs)
                    or len(pairs) < int(bootstrap.minimum_pairs)
                    or not grouped
                ):
                    estimates.append(float("nan"))
                    lows.append(float("nan"))
                    highs.append(float("nan"))
                    continue
                observations = [
                    Observation(
                        seed=int(seed),
                        graph=int(graph),
                        source=int(source),
                        donor=int(donor),
                        value=np.asarray(
                            [np.sum(values), len(values)],
                            dtype=np.float64,
                        ),
                    )
                    for (graph, source, donor), values in sorted(grouped.items())
                ]

                def graph_reduce(values: np.ndarray) -> np.ndarray:
                    return trimmed_mean(
                        values[:, 0] / values[:, 1],
                        bootstrap.trim_fraction,
                        axis=0,
                    )

                offset += 1
                interval = nested_percentile_interval(
                    observations,
                    dataclasses.replace(
                        bootstrap,
                        rng_seed=int(bootstrap.rng_seed) + 90_000 + offset,
                        resample_source=False,
                    ),
                    graph_reduce=graph_reduce,
                )
                estimates.append(float(interval.estimate))
                lows.append(float(interval.low))
                highs.append(float(interval.high))
            if not any(np.isfinite(estimates)):
                continue
            output[(channel, role)] = {
                "labels": axis.labels,
                "estimate": np.asarray(estimates, dtype=np.float64),
                "low": np.asarray(lows, dtype=np.float64),
                "high": np.asarray(highs, dtype=np.float64),
                "graphs": np.asarray(graph_support, dtype=np.int64),
                "pairs": np.asarray(pair_support, dtype=np.int64),
                "events": np.asarray(event_support, dtype=np.int64),
            }
    return output


def plot_role_conditioned_carriage(
    inputs: PaperInputs,
    *,
    output_dir: Path,
    models: Sequence[str],
    seeds: Sequence[int],
    records: int,
    bootstrap: BootstrapPolicy,
    headline: bool,
) -> None:
    """Supporting NAR-specific carriage mechanism, best validation seed per model."""

    import matplotlib.pyplot as plt

    profiles: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    table: list[dict[str, Any]] = []
    all_distances = [
        int(float(row["distance"]))
        for model in models
        for channel in ("semantic", "structural")
        for row in inputs.carriage.get((str(model), int(records)), {})
        .get("channels", {})
        .get(channel, {})
        .get("pairs", ())
        if np.isfinite(_as_float(row.get("distance")))
    ]
    if not all_distances:
        return
    shared_distance_axis = display_bins(
        tuple(range(max(all_distances) + 1)),
        max_points=14,
    )
    for model in models:
        key = (str(model), int(records))
        if key not in inputs.carriage or key not in inputs.best_seeds:
            continue
        seed = int(inputs.best_seeds[key])
        profiles[str(model)] = role_conditioned_carriage_profiles(
            inputs.carriage[key],
            seed=seed,
            bootstrap=bootstrap,
            distance_axis=shared_distance_axis,
        )
        for (channel, role), result in profiles[str(model)].items():
            for position, label in enumerate(result["labels"]):
                table.append(
                    {
                        "model": model,
                        "N": int(records),
                        "best_validation_seed": seed,
                        "channel": channel,
                        "source_role": role,
                        "distance_group": label,
                        "functional_carriage": float(
                            result["estimate"][position]
                        ),
                        "ci95_low": float(result["low"][position]),
                        "ci95_high": float(result["high"][position]),
                        "events": int(result["events"][position]),
                        "graphs": int(result["graphs"][position]),
                        "eligible_pairs": int(result["pairs"][position]),
                    }
                )
    if not profiles:
        return
    _write_csv(
        output_dir
        / "tables"
        / (
            f"role_conditioned_carriage_N{records}.csv"
            if headline
            else f"supplement_role_conditioned_carriage_N{records}.csv"
        ),
        table,
    )
    with _style():
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(7.5, 5.25),
            constrained_layout=False,
            sharey=True,
            sharex=True,
            squeeze=False,
        )
        for row_index, channel in enumerate(("semantic", "structural")):
            for column, role in enumerate(("query", "record")):
                axis = axes[row_index, column]
                for model in models:
                    result = profiles.get(str(model), {}).get((channel, role))
                    if result is None:
                        continue
                    x = np.arange(len(result["labels"]))
                    estimate = np.asarray(result["estimate"])
                    axis.plot(
                        x,
                        estimate,
                        color=MODEL_COLOURS[str(model)],
                        marker=MODEL_MARKERS[str(model)],
                        linewidth=1.35,
                        markersize=4,
                        label=MODEL_LABELS[str(model)],
                    )
                    axis.fill_between(
                        x,
                        result["low"],
                        result["high"],
                        color=MODEL_COLOURS[str(model)],
                        alpha=0.13,
                        linewidth=0,
                    )
                axis.set_xticks(np.arange(len(shared_distance_axis.labels)))
                axis.set_xticklabels(
                    shared_distance_axis.labels,
                    rotation=35,
                    ha="right",
                    rotation_mode="anchor",
                    fontsize=6.6,
                )
                axis.set_title(
                    f"{channel.capitalize()} donor swap\n"
                    + (
                        "Query source"
                        if role == "query"
                        else "Requested-record source"
                    ),
                    fontsize=8.2,
                    color=(
                        SEMANTIC_COLOUR
                        if channel == "semantic"
                        else STRUCTURAL_COLOUR
                    ),
                    pad=6,
                )
        handles, _ = axes[0, 0].get_legend_handles_labels()
        labels = []
        for model in models:
            mean, sd, _ = accuracy_summary(
                inputs.performance,
                str(model),
                int(records),
                seeds,
            )
            accuracy = (
                rf"held-out acc. ${mean:.2f}\pm{sd:.2f}$"
                if np.isfinite(mean)
                else "held-out accuracy unavailable"
            )
            labels.append(f"{MODEL_LABELS[str(model)]}\n{accuracy}")
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.006),
            ncol=len(handles),
            fontsize=7.0,
            handlelength=2.0,
            columnspacing=1.8,
        )
        fig.suptitle(
            rf"Role-conditioned Functional carriage at $N={records}$",
            fontsize=13,
            y=0.98,
        )
        fig.supylabel(
            r"Functional carriage  $F_{\rm sens}$",
            fontsize=9.3,
            x=0.02,
        )
        fig.supxlabel(
            "Shortest-path distance from source",
            fontsize=9.3,
            y=0.135,
        )
        fig.subplots_adjust(
            left=0.10,
            right=0.99,
            bottom=0.22,
            top=0.82,
            hspace=0.50,
            wspace=0.25,
        )
        directory = (
            output_dir / "figures"
            if headline
            else output_dir / "supplementary" / "carriage"
        )
        _save_figure(
            fig,
            directory,
            CARRIAGE_FIGURE_STEM.format(N=int(records)),
            {
                "paper_version": PAPER_VERSION,
                "N": int(records),
                "seed_policy": "lowest-validation-loss seed independently per model",
                "estimand": "raw Functional carriage F_sens; beneficial carriage absent",
                "uncertainty": "95% nested graph/donor percentile interval within selected seed",
                "distance_groups": (
                    "one shared axis across models/channels; at most 14 contiguous groups"
                ),
                "reporting_floor": (
                    f"{bootstrap.minimum_graphs} graphs and "
                    f"{bootstrap.minimum_pairs} eligible carrier/source pairs"
                ),
            },
        )
        plt.close(fig)


def make_paper_figures(
    inputs: PaperInputs,
    *,
    output_dir: Path,
    models: Sequence[str],
    seeds: Sequence[int],
    score_ns: Sequence[int],
    causal_ns: Sequence[int],
    counterfactual_ns: Sequence[int],
    carriage_ns: Sequence[int],
    performance_ns: Sequence[int],
    headline_n: int,
    supplement_ns: Sequence[int],
    bootstrap: BootstrapPolicy,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_core_specialisation(
        inputs,
        output_dir=output_dir,
        models=models,
        seeds=seeds,
        records=int(headline_n),
        headline=True,
    )
    for records in supplement_ns:
        if int(records) == int(headline_n) or int(records) not in set(score_ns):
            continue
        plot_core_specialisation(
            inputs,
            output_dir=output_dir,
            models=models,
            seeds=seeds,
            records=int(records),
            headline=False,
        )

    causal_rows = causal_summary_rows(
        inputs,
        models=models,
        causal_ns=causal_ns,
        seeds=seeds,
    )
    _write_csv(output_dir / "tables" / "core_causal_validation.csv", causal_rows)
    plot_core_causal_validation(
        inputs,
        output_dir=output_dir,
        models=models,
        seeds=seeds,
        records=int(headline_n),
        all_rows=causal_rows,
    )

    conditional_rows = conditional_family_rows(
        inputs,
        models=models,
        seeds=seeds,
        records=int(headline_n),
    )
    _write_csv(
        output_dir / "tables" / "conditional_core_family_fingerprints.csv",
        conditional_rows,
    )
    counterfactual_rows = counterfactual_double_dissociation_rows(
        inputs.counterfactual,
        models=models,
        ns=counterfactual_ns,
        bootstrap=bootstrap,
    )
    _write_csv(
        output_dir / "tables" / "counterfactual_double_dissociation.csv",
        counterfactual_rows,
    )
    plot_role_and_counterfactual(
        inputs,
        output_dir=output_dir,
        models=models,
        seeds=seeds,
        records=int(headline_n),
        counterfactual_ns=counterfactual_ns,
        conditional_rows=conditional_rows,
        counterfactual_rows=counterfactual_rows,
    )
    plot_competence_and_causal_grounding(
        inputs,
        output_dir=output_dir,
        models=models,
        performance_ns=performance_ns,
        causal_ns=causal_ns,
        seeds=seeds,
        causal_rows=causal_rows,
    )
    for records in carriage_ns:
        plot_role_conditioned_carriage(
            inputs,
            output_dir=output_dir,
            models=models,
            seeds=seeds,
            records=int(records),
            bootstrap=bootstrap,
            headline=int(records) == int(headline_n),
        )

    figures = sorted(
        str(path.relative_to(output_dir))
        for path in output_dir.rglob("*.png")
    )
    tables = sorted(
        str(path.relative_to(output_dir))
        for path in output_dir.rglob("*.csv")
    )
    atomic_json(
        output_dir / "paper_figure_index.json",
        {
            "paper_version": PAPER_VERSION,
            "source_policy": (
                "canonical and v1 extension artifacts opened read-only; no checkpoint inference"
            ),
            "headline_figures": [
                CORE_FIGURE_STEM.format(N=int(headline_n)),
                CAUSAL_FIGURE_STEM.format(N=int(headline_n)),
                INTERPRETATION_FIGURE_STEM,
                TRANSITION_FIGURE_STEM,
            ],
            "figures": figures,
            "tables": tables,
            "N_values": {
                "scores": list(map(int, score_ns)),
                "causal": list(map(int, causal_ns)),
                "counterfactual": list(map(int, counterfactual_ns)),
                "carriage": list(map(int, carriage_ns)),
                "performance": list(map(int, performance_ns)),
            },
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cache-only paper synthesis for the fixed-N NAR methodology experiment"
    )
    parser.add_argument("--phase", choices=("figures",), default="figures")
    parser.add_argument(
        "--drive-root",
        default="/content/drive/MyDrive/graph_specialisation_metrics/nar_grit",
    )
    parser.add_argument("--training-run-name", default="nar_grit_fixed_n_v3")
    parser.add_argument("--base-analysis-name", default="canonical_nar_analysis_d128")
    parser.add_argument("--source-extension-name", default=SOURCE_EXTENSION_NAME)
    parser.add_argument("--paper-analysis-name", default=DEFAULT_PAPER_ANALYSIS_NAME)
    parser.add_argument("--models", default="1hop,2hop,dense")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--cached-ns", default="4,16,64")
    parser.add_argument("--score-ns", default="4,8,16,32,64")
    parser.add_argument("--causal-ns", default="4,16,64")
    parser.add_argument("--counterfactual-ns", default="4,8,16")
    parser.add_argument("--carriage-ns", default="4,16,64")
    parser.add_argument("--performance-ns", default="4,8,16,32,64,80")
    parser.add_argument("--headline-n", type=int, default=16)
    parser.add_argument("--supplement-ns", default="4,8,32,64")
    parser.add_argument("--analysis-width", type=int, default=128)
    parser.add_argument("--counterfactual-donors-per-role", type=int, default=8)
    parser.add_argument("--accuracy-gate", type=float, default=0.85)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    del args.phase
    models = _parse_csv_strings(args.models)
    seeds = _parse_csv_ints(args.seeds)
    cached_ns = _parse_csv_ints(args.cached_ns)
    score_ns = _parse_csv_ints(args.score_ns)
    causal_ns = _parse_csv_ints(args.causal_ns)
    counterfactual_ns = _parse_csv_ints(args.counterfactual_ns)
    carriage_ns = _parse_csv_ints(args.carriage_ns)
    performance_ns = _parse_csv_ints(args.performance_ns)
    supplement_ns = _parse_csv_ints(args.supplement_ns)
    if any(model not in MODEL_ORDER for model in models):
        raise ValueError(f"models must be drawn from {MODEL_ORDER}")
    if len(seeds) != 3:
        raise ValueError("the registered paper synthesis requires exactly three seeds")
    if int(args.headline_n) not in set(score_ns):
        raise ValueError("headline-n must be present in score-ns")
    if int(args.headline_n) not in set(causal_ns):
        raise ValueError("headline-n must be present in causal-ns")
    if not set(causal_ns).issubset(set(cached_ns)):
        raise ValueError("causal-ns must refer to the completed canonical analysis cells")
    if not set(counterfactual_ns).issubset(set(score_ns)):
        raise ValueError("counterfactual-ns must be a subset of score-ns")
    if not set(carriage_ns).issubset(set(cached_ns)):
        raise ValueError("carriage-ns must refer to the completed canonical analysis cells")

    drive_root = Path(args.drive_root)
    training_run_dir = drive_root / str(args.training_run_name)
    base_analysis_root = training_run_dir / str(args.base_analysis_name)
    base_canonical_root = base_analysis_root / "canonical"
    source_extension_root = (
        base_analysis_root / "extensions" / str(args.source_extension_name)
    )
    output_dir = base_analysis_root / "extensions" / str(args.paper_analysis_name)
    _safe_extension_layout(base_analysis_root, output_dir)
    if output_dir.resolve() == source_extension_root.resolve():
        raise ValueError("paper-analysis-name must differ from source-extension-name")

    inputs, policies = load_paper_inputs(
        base_analysis_root=base_analysis_root,
        base_canonical_root=base_canonical_root,
        source_extension_root=source_extension_root,
        training_run_dir=training_run_dir,
        models=models,
        score_ns=score_ns,
        causal_ns=causal_ns,
        counterfactual_ns=counterfactual_ns,
        carriage_ns=carriage_ns,
        cached_ns=cached_ns,
        performance_ns=performance_ns,
        seeds=seeds,
        width=int(args.analysis_width),
        donors_per_role=int(args.counterfactual_donors_per_role),
        accuracy_gate=float(args.accuracy_gate),
    )
    _, _, bootstrap, _, _ = policies
    make_paper_figures(
        inputs,
        output_dir=output_dir,
        models=models,
        seeds=seeds,
        score_ns=score_ns,
        causal_ns=causal_ns,
        counterfactual_ns=counterfactual_ns,
        carriage_ns=carriage_ns,
        performance_ns=performance_ns,
        headline_n=int(args.headline_n),
        supplement_ns=supplement_ns,
        bootstrap=bootstrap,
    )
    atomic_json(
        output_dir / "run_summary.json",
        {
            "paper_version": PAPER_VERSION,
            "base_analysis_root": str(base_analysis_root),
            "base_canonical_root": str(base_canonical_root),
            "source_extension_root": str(source_extension_root),
            "output_dir": str(output_dir),
            "cache_policy": (
                "all source artifacts read-only; this frontend writes derived tables, metadata, "
                "and figures only"
            ),
            "checkpoint_inference": False,
            "models": list(models),
            "seeds": list(seeds),
            "score_N_values": list(score_ns),
            "causal_N_values": list(causal_ns),
            "counterfactual_N_values": list(counterfactual_ns),
            "carriage_N_values": list(carriage_ns),
            "performance_N_values": list(performance_ns),
        },
    )
    print(f"[done] paper figures and tables saved under {output_dir}", flush=True)
    return {"output_dir": str(output_dir)}


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    return run(build_parser().parse_args(list(argv) if argv is not None else None))


__all__ = [
    "CAUSAL_FIGURE_STEM",
    "CORE_FIGURE_STEM",
    "DEFAULT_PAPER_ANALYSIS_NAME",
    "INTERPRETATION_FIGURE_STEM",
    "PAPER_VERSION",
    "PaperInputs",
    "TRANSITION_FIGURE_STEM",
    "accuracy_summary",
    "build_parser",
    "causal_summary_rows",
    "conditional_family_rows",
    "conditional_source_scores",
    "counterfactual_double_dissociation_rows",
    "counterfactual_graph_observations",
    "main",
    "role_conditioned_carriage_profiles",
    "run",
]


if __name__ == "__main__":
    main()
