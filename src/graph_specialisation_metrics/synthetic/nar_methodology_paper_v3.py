"""Publication-focused NAR synthesis with complete capacity and validity analyses.

This is a model-free frontend.  It reads canonical N=4/16/64 artifacts, protected N=8/32
transition scores, protected N=8/32 causal-completion artifacts, and the v1 role/counterfactual
extension.  All derived outputs are written to a new v3 namespace.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..methodology.bootstrap import Observation, nested_percentile_interval
from ..methodology.cache import atomic_json
from ..methodology.protocol import BootstrapPolicy
from . import nar_methodology_paper as v2
from .nar_canonical_analysis import (
    MODEL_COLOURS,
    MODEL_LABELS,
    MODEL_MARKERS,
    MODEL_ORDER,
    SEED_MARKERS,
    _save_figure,
    _style,
    _write_csv,
)
from .nar_causal_transition import (
    DEFAULT_CAUSAL_EXTENSION_NAME,
    load_transition_causal_artifact,
)
from .nar_methodology_extension import (
    DEFAULT_EXTENSION_NAME as SOURCE_EXTENSION_NAME,
    _methodology_policies,
    _parse_csv_ints,
    _parse_csv_strings,
    _safe_extension_layout,
)


PAPER_VERSION = "nar-methodology-paper-v3"
DEFAULT_PAPER_ANALYSIS_NAME = "nar_methodology_paper_v3"

CAUSAL_FIGURE_STEM = "02_core_causal_validation_N{N}"
INTERPRETATION_FIGURE_STEM = "03_query_localisation_and_counterfactual_validation"
TRANSITION_FIGURE_STEM = "04_competence_and_causal_grounding"
ORGANISATION_FIGURE_STEM = "S04_head_organisation_across_capacity"
OVERLAP_FIGURE_STEM = "S05_core_role_family_overlap"
DISCRIMINANT_FIGURE_STEM = "S06_channel_discriminant_validity"
FINGERPRINT_FIGURE_STEM = "S03_conditional_source_fingerprint_N{N}"


def load_v3_inputs(
    *,
    base_analysis_root: Path,
    base_canonical_root: Path,
    source_extension_root: Path,
    causal_extension_root: Path,
    training_run_dir: Path,
    models: Sequence[str],
    score_ns: Sequence[int],
    canonical_causal_ns: Sequence[int],
    transition_causal_ns: Sequence[int],
    counterfactual_ns: Sequence[int],
    carriage_ns: Sequence[int],
    cached_ns: Sequence[int],
    performance_ns: Sequence[int],
    seeds: Sequence[int],
    width: int,
    donors_per_role: int,
    accuracy_gate: float,
) -> tuple[v2.PaperInputs, tuple[Any, ...]]:
    inputs, policies = v2.load_paper_inputs(
        base_analysis_root=base_analysis_root,
        base_canonical_root=base_canonical_root,
        source_extension_root=source_extension_root,
        training_run_dir=training_run_dir,
        models=models,
        score_ns=score_ns,
        causal_ns=canonical_causal_ns,
        counterfactual_ns=counterfactual_ns,
        carriage_ns=carriage_ns,
        cached_ns=cached_ns,
        performance_ns=performance_ns,
        seeds=seeds,
        width=int(width),
        donors_per_role=int(donors_per_role),
        accuracy_gate=float(accuracy_gate),
    )
    causal = dict(inputs.causal)
    for records in transition_causal_ns:
        for model in models:
            for seed in seeds:
                binding = inputs.score_bindings[
                    (str(model), int(records), int(seed))
                ]
                causal[(str(model), int(records), int(seed))] = (
                    load_transition_causal_artifact(
                        extension_root=causal_extension_root,
                        binding=binding,
                    )
                )
    return dataclasses.replace(inputs, causal=causal), policies


def plot_compact_causal_validation(
    inputs: v2.PaperInputs,
    *,
    output_dir: Path,
    models: Sequence[str],
    seeds: Sequence[int],
    records: int,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    """Two-row headline validation; unstable family interactions are kept out."""

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
            2,
            len(models),
            figsize=(3.05 * len(models), 5.25),
            squeeze=False,
            gridspec_kw={"height_ratios": (1.0, 1.15)},
        )
        for column, model in enumerate(models):
            selected = [
                row
                for row in rows
                if row["model"] == model and int(row["N"]) == int(records)
            ]
            raw = [row for row in selected if row["section"] == "raw_score_calibration"]
            v2._scatter_seed_summary(
                axes[0, column],
                raw,
                raw_labels,
                seeds=seeds,
                colour_by_label={
                    raw_labels[0]: v2.SEMANTIC_COLOUR,
                    raw_labels[1]: v2.STRUCTURAL_COLOUR,
                    raw_labels[2]: v2.CONTROL_COLOUR,
                    raw_labels[3]: v2.CONTROL_COLOUR,
                },
            )
            coordinates = [
                row for row in selected if row["section"] == "coordinate_validation"
            ]
            v2._scatter_seed_summary(
                axes[1, column],
                coordinates,
                coordinate_labels,
                seeds=seeds,
                colour_by_label={
                    label: ("#8A5A2B" if label.startswith("$J") else "#76528B")
                    for label in coordinate_labels
                },
            )
            for axis in axes[:, column]:
                axis.set_xlim(-1.05, 1.05)
                axis.set_xlabel(r"Spearman $\rho$")
            axes[0, column].set_title(
                f"{MODEL_LABELS[str(model)]}\n"
                + v2._accuracy_subtitle(
                    inputs.performance, str(model), int(records), seeds
                ),
                fontsize=10.2,
                pad=8,
            )
            if column:
                for axis in axes[:, column]:
                    axis.set_yticklabels([])
                    axis.tick_params(axis="y", length=0)
        axes[0, 0].set_ylabel("Raw-score\ncalibration")
        axes[1, 0].set_ylabel("Coordinate\nvalidation")
        seed_handles = [
            Line2D(
                [], [], marker=SEED_MARKERS[index], linestyle="none",
                markerfacecolor="#666666", markeredgecolor="white",
                label=f"Seed {seed}", markersize=5.5,
            )
            for index, seed in enumerate(seeds)
        ]
        fig.legend(
            handles=seed_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.008),
            ncol=len(seed_handles),
            fontsize=7.4,
        )
        fig.suptitle(
            rf"Canonical scores predict held-out causal function at $N={records}$",
            fontsize=13,
            y=0.985,
        )
        fig.subplots_adjust(
            left=0.13, right=0.99, bottom=0.13, top=0.86,
            hspace=0.48, wspace=0.24,
        )
        _save_figure(
            fig,
            output_dir / "figures",
            CAUSAL_FIGURE_STEM.format(N=int(records)),
            {
                "paper_version": PAPER_VERSION,
                "N": int(records),
                "family_interaction": "excluded from headline; retained in diagnostic table",
                "replication": "seed points plus an unfilled descriptive mean diamond",
            },
        )
        plt.close(fig)


def localisation_rows(
    conditional_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Bounded query-versus-record allocation on the unconditional core scale."""

    grouped: dict[tuple[str, int, int, str], dict[str, float]] = {}
    for row in conditional_rows:
        key = (
            str(row["model"]),
            int(row["N"]),
            int(row["seed"]),
            str(row["family"]),
        )
        grouped.setdefault(key, {})[str(row["component"])] = float(row["estimate"])
    output: list[dict[str, Any]] = []
    for (model, records, seed, family), values in sorted(grouped.items()):
        channel = "semantic" if family == "semantic_leaning" else "structural"
        query = values.get(f"{channel}_query", np.nan)
        record = values.get(f"{channel}_record", np.nan)
        denominator = abs(query) + abs(record)
        contrast = (
            float((query - record) / denominator)
            if np.isfinite(denominator) and denominator > 1e-12
            else float("nan")
        )
        output.append(
            {
                "model": model,
                "N": records,
                "seed": seed,
                "family": family,
                "channel": channel,
                "query_sensitivity": query,
                "record_sensitivity": record,
                "query_localisation": contrast,
                "definition": "(query-record)/(|query|+|record|)",
                "normalisation": "unconditional core channel mean",
            }
        )
    return output


def pooled_counterfactual_rows(
    results: Mapping[tuple[str, int, int], Mapping[str, Any]],
    *,
    models: Sequence[str],
    ns: Sequence[int],
    bootstrap: BootstrapPolicy,
) -> list[dict[str, Any]]:
    """Pool graph-level exact-recall double differences over competent N cells."""

    rows: list[dict[str, Any]] = []
    for model_index, model in enumerate(models):
        observations: list[Observation] = []
        support_graphs = 0
        contributing_ns: list[int] = []
        for n_index, records in enumerate(ns):
            cell, support = v2.counterfactual_graph_observations(
                results,
                model=str(model),
                records=int(records),
                require_correct=True,
            )
            if not cell:
                continue
            contributing_ns.append(int(records))
            support_graphs += int(support["graphs"])
            observations.extend(
                Observation(
                    seed=int(item.seed),
                    graph=(n_index + 1) * 1_000_000 + int(item.graph),
                    source=int(item.source),
                    donor=int(item.donor),
                    value=np.asarray(item.value, dtype=np.float64),
                )
                for item in cell
            )
        if not observations:
            continue
        interval = nested_percentile_interval(
            observations,
            dataclasses.replace(
                bootstrap,
                rng_seed=int(bootstrap.rng_seed) + 91_000 + model_index,
                resample_source=False,
                resample_donor=False,
            ),
            transform=v2._counterfactual_vector_transform,
        )
        rows.append(
            {
                "model": model,
                "N": "pooled",
                "analysis_set": "successful_exact_recall",
                "estimand": "adjusted",
                "estimate": float(interval.estimate[2]),
                "ci95_low": float(interval.low[2]),
                "ci95_high": float(interval.high[2]),
                "graphs": support_graphs,
                "seeds": len({item.seed for item in observations}),
                "pooled_N_values": ",".join(map(str, contributing_ns)),
                "estimator": "hierarchical seed/graph bootstrap over competent cells",
            }
        )
    return rows


def plot_localisation_and_counterfactual(
    inputs: v2.PaperInputs,
    *,
    output_dir: Path,
    models: Sequence[str],
    seeds: Sequence[int],
    score_ns: Sequence[int],
    counterfactual_rows: Sequence[Mapping[str, Any]],
    localisation: Sequence[Mapping[str, Any]],
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    family_style = {
        "semantic_leaning": (v2.SEMANTIC_COLOUR, "Semantic family"),
        "structural_leaning": (v2.STRUCTURAL_COLOUR, "Structural family"),
    }
    finite = [
        float(row[key])
        for row in counterfactual_rows
        if row["analysis_set"] == "successful_exact_recall"
        and row["estimand"] == "adjusted"
        for key in ("ci95_low", "ci95_high")
        if np.isfinite(v2._as_float(row.get(key)))
    ]
    limits = (
        min(min(finite, default=-0.1), 0) - 0.05,
        max(max(finite, default=0.1), 0) + 0.05,
    )
    with _style():
        fig, axes = plt.subplots(
            2,
            len(models),
            figsize=(3.15 * len(models), 5.75),
            squeeze=False,
            gridspec_kw={"height_ratios": (1.08, 0.92)},
        )
        for column, model in enumerate(models):
            for family, (colour, _) in family_style.items():
                seed_matrix = []
                for seed_index, seed in enumerate(seeds):
                    values = []
                    for records in score_ns:
                        match = [
                            row for row in localisation
                            if row["model"] == model
                            and int(row["N"]) == int(records)
                            and int(row["seed"]) == int(seed)
                            and row["family"] == family
                        ]
                        values.append(
                            float(match[0]["query_localisation"]) if match else np.nan
                        )
                    seed_matrix.append(values)
                    axes[0, column].plot(
                        score_ns, values, color=colour, alpha=0.24, linewidth=0.7
                    )
                    axes[0, column].scatter(
                        score_ns, values, color=colour,
                        marker=SEED_MARKERS[seed_index], s=18,
                        edgecolor="white", linewidth=0.35, zorder=3,
                    )
                mean = np.nanmean(np.asarray(seed_matrix, dtype=np.float64), axis=0)
                axes[0, column].plot(
                    score_ns, mean, color=colour, linewidth=1.6, zorder=2
                )
            axes[0, column].axhline(
                0, color="#888888", linestyle="--", linewidth=0.7
            )
            axes[0, column].set_ylim(-1.05, 1.05)
            axes[0, column].set_xticks(score_ns)
            tick_labels = []
            for records in score_ns:
                accuracy, _, _ = v2.accuracy_summary(
                    inputs.performance, str(model), int(records), seeds
                )
                tick_labels.append(
                    f"{records}\n{accuracy:.2f}" if np.isfinite(accuracy) else f"{records}\n—"
                )
            axes[0, column].set_xticklabels(tick_labels, fontsize=6.6)
            axes[0, column].set_xlabel("$N$  (accuracy below)")
            axes[0, column].set_title(MODEL_LABELS[str(model)], fontsize=10.2)
            if column == 0:
                axes[0, column].set_ylabel(
                    "Query localisation\n"
                    r"$(S_q-S_r)/(|S_q|+|S_r|)$"
                )

            forest = [
                row for row in counterfactual_rows
                if row["model"] == model
                and row["analysis_set"] == "successful_exact_recall"
                and row["estimand"] == "adjusted"
            ]
            cells = sorted(
                [row for row in forest if row["N"] != "pooled"],
                key=lambda row: int(row["N"]),
            )
            pooled = [row for row in forest if row["N"] == "pooled"]
            ordered = [*cells, *pooled]
            for position, row in enumerate(ordered):
                axes[1, column].plot(
                    [float(row["ci95_low"]), float(row["ci95_high"])],
                    [position, position],
                    color=MODEL_COLOURS[str(model)],
                    linewidth=1.15 if row["N"] != "pooled" else 1.7,
                )
                axes[1, column].scatter(
                    float(row["estimate"]), position,
                    marker="D" if row["N"] == "pooled" else MODEL_MARKERS[str(model)],
                    s=42 if row["N"] == "pooled" else 30,
                    color=MODEL_COLOURS[str(model)],
                    edgecolor="white", linewidth=0.5, zorder=3,
                )
            axes[1, column].axvline(
                0, color="#888888", linestyle="--", linewidth=0.7
            )
            axes[1, column].set_xlim(*limits)
            axes[1, column].set_yticks(range(len(ordered)))
            axes[1, column].set_yticklabels(
                [
                    (
                        "Pooled"
                        if row["N"] == "pooled"
                        else rf"$N={int(row['N'])}$"
                    )
                    for row in ordered
                ],
                fontsize=7,
            )
            if ordered:
                axes[1, column].set_ylim(-0.55, len(ordered) - 0.45)
            if column == 0:
                axes[1, column].set_ylabel("Exact-recall\ncounterfactual test")
        handles = [
            Line2D([], [], color=colour, linewidth=1.6, label=label)
            for colour, label in family_style.values()
        ] + [
            Line2D(
                [], [], marker=SEED_MARKERS[index], linestyle="none",
                color="#666666", label=f"Seed {seed}", markersize=5,
            )
            for index, seed in enumerate(seeds)
        ]
        fig.legend(
            handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.008),
            ncol=len(handles), fontsize=7.2,
        )
        fig.text(
            0.55,
            0.095,
            "Control-adjusted counterfactual double dissociation",
            ha="center",
            va="center",
            fontsize=8.8,
        )
        fig.suptitle(
            "Core specialisation localises retrieval and supports exact counterfactual recall",
            fontsize=12.8,
            y=0.985,
        )
        fig.subplots_adjust(
            left=0.10, right=0.99, bottom=0.15, top=0.90,
            hspace=0.40, wspace=0.34,
        )
        _save_figure(
            fig,
            output_dir / "figures",
            INTERPRETATION_FIGURE_STEM,
            {
                "paper_version": PAPER_VERSION,
                "localisation": (
                    "descriptive conditional source allocation on the unconditional core scale"
                ),
                "counterfactual": (
                    "accuracy-gated exact-answer events; family double dissociation minus "
                    "matched-control double dissociation"
                ),
                "pooled_counterfactual": "hierarchical seed/graph interval across competent cells",
            },
        )
        plt.close(fig)


def plot_conditional_fingerprint_supplement(
    inputs: v2.PaperInputs,
    *,
    output_dir: Path,
    models: Sequence[str],
    seeds: Sequence[int],
    records: int,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    import matplotlib.pyplot as plt

    with _style():
        fig, axes = plt.subplots(
            1, len(models), figsize=(3.0 * len(models), 2.75), squeeze=False
        )
        x = np.arange(len(v2.CONDITIONAL_COMPONENTS))
        for column, model in enumerate(models):
            axis = axes[0, column]
            for family, colour in (
                ("semantic_leaning", v2.SEMANTIC_COLOUR),
                ("structural_leaning", v2.STRUCTURAL_COLOUR),
            ):
                for seed_index, seed in enumerate(seeds):
                    values = []
                    for component in v2.CONDITIONAL_COMPONENTS:
                        match = [
                            row for row in rows
                            if row["model"] == model
                            and int(row["N"]) == int(records)
                            and int(row["seed"]) == int(seed)
                            and row["family"] == family
                            and row["component"] == component
                        ]
                        values.append(float(match[0]["estimate"]) if match else np.nan)
                    axis.plot(x, values, color=colour, alpha=0.28, linewidth=0.7)
                    axis.scatter(
                        x, values, color=colour,
                        marker=SEED_MARKERS[seed_index], s=20,
                        edgecolor="white", linewidth=0.35,
                    )
            axis.axvline(1.5, color="#999999", linestyle="--", linewidth=0.65)
            axis.set_xticks(x)
            axis.set_xticklabels(v2.CONDITIONAL_LABELS, fontsize=6.5)
            axis.set_title(
                f"{MODEL_LABELS[str(model)]}\n"
                + v2._accuracy_subtitle(
                    inputs.performance, str(model), int(records), seeds
                ),
                fontsize=9,
            )
            if column == 0:
                axis.set_ylabel("Conditional sensitivity")
        fig.suptitle(
            rf"Conditional source fingerprint at $N={records}$",
            fontsize=11.5,
            y=0.99,
        )
        fig.subplots_adjust(
            left=0.09, right=0.99, bottom=0.19, top=0.78, wspace=0.32
        )
        _save_figure(
            fig,
            output_dir / "supplementary" / "roles",
            FINGERPRINT_FIGURE_STEM.format(N=int(records)),
            {
                "paper_version": PAPER_VERSION,
                "status": "supplementary diagnostic; replaced in main text by localisation",
            },
        )
        plt.close(fig)


def organisation_rows(
    inputs: v2.PaperInputs,
    *,
    models: Sequence[str],
    seeds: Sequence[int],
    ns: Sequence[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for records in ns:
        heads = v2._head_interval_rows(
            inputs, models=models, seeds=seeds, records=int(records)
        )
        for model in models:
            for seed in seeds:
                cell = [
                    row for row in heads
                    if row["model"] == model and int(row["seed"]) == int(seed)
                ]
                if not cell:
                    continue
                active = [row for row in cell if bool(row["active"])]
                j = np.maximum(
                    np.asarray([float(row["J"]) for row in cell], dtype=np.float64),
                    0,
                )
                top_count = max(1, int(np.ceil(0.20 * len(j))))
                total = float(np.sum(j))
                top_share = (
                    float(np.sum(np.sort(j)[-top_count:]) / total)
                    if total > 0 else float("nan")
                )
                abs_d = np.asarray(
                    [abs(float(row["D_rel"])) for row in active],
                    dtype=np.float64,
                )
                rows.append(
                    {
                        "model": model,
                        "N": int(records),
                        "seed": int(seed),
                        "heads": len(cell),
                        "active_heads": len(active),
                        "active_fraction": len(active) / len(cell),
                        "top20_J_share": top_share,
                        "median_active_abs_D_rel": (
                            float(np.median(abs_d)) if len(abs_d) else float("nan")
                        ),
                        "semantic_family_heads": sum(
                            row["family"] == "semantic_leaning" for row in cell
                        ),
                        "structural_family_heads": sum(
                            row["family"] == "structural_leaning" for row in cell
                        ),
                        "cross_N_head_alignment": False,
                    }
                )
    return rows


def plot_organisation(
    rows: Sequence[Mapping[str, Any]],
    *,
    output_dir: Path,
    models: Sequence[str],
    seeds: Sequence[int],
    ns: Sequence[int],
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    metrics = (
        ("active_fraction", "Active-head fraction"),
        ("top20_J_share", "Top-20% share of $J$"),
        ("median_active_abs_D_rel", r"Median active $|D_{rel}|$"),
        ("family_heads", "Frozen family heads"),
    )
    with _style():
        fig, axes = plt.subplots(1, 4, figsize=(10.5, 2.9))
        for axis, (metric, title) in zip(axes, metrics):
            for model in models:
                if metric == "family_heads":
                    for family_key, linestyle in (
                        ("semantic_family_heads", "-"),
                        ("structural_family_heads", "--"),
                    ):
                        means = []
                        for records in ns:
                            values = [
                                float(row[family_key]) for row in rows
                                if row["model"] == model and int(row["N"]) == int(records)
                            ]
                            means.append(float(np.mean(values)) if values else np.nan)
                        axis.plot(
                            ns, means, color=MODEL_COLOURS[str(model)],
                            linestyle=linestyle, linewidth=1.25,
                            marker=MODEL_MARKERS[str(model)], markersize=3.5,
                        )
                else:
                    seed_values = []
                    for seed_index, seed in enumerate(seeds):
                        values = []
                        for records in ns:
                            match = [
                                row for row in rows
                                if row["model"] == model
                                and int(row["N"]) == int(records)
                                and int(row["seed"]) == int(seed)
                            ]
                            values.append(float(match[0][metric]) if match else np.nan)
                        seed_values.append(values)
                        axis.plot(
                            ns, values, color=MODEL_COLOURS[str(model)],
                            alpha=0.20, linewidth=0.65,
                        )
                    axis.plot(
                        ns, np.nanmean(np.asarray(seed_values), axis=0),
                        color=MODEL_COLOURS[str(model)],
                        marker=MODEL_MARKERS[str(model)],
                        linewidth=1.35, markersize=3.7,
                        label=MODEL_LABELS[str(model)],
                    )
            axis.set_xticks(ns)
            axis.set_xlabel("Memory size, $N$")
            axis.set_title(title, fontsize=9.1)
        axes[0].set_ylim(0, 1.02)
        axes[1].set_ylim(0, 1.02)
        handles, labels = axes[0].get_legend_handles_labels()
        family_handles = [
            Line2D(
                [], [], color="#444444", linestyle="-",
                label="Semantic family count",
            ),
            Line2D(
                [], [], color="#444444", linestyle="--",
                label="Structural family count",
            ),
        ]
        fig.legend(
            [*handles, *family_handles],
            [*labels, *(item.get_label() for item in family_handles)],
            loc="lower center", bbox_to_anchor=(0.5, 0.005),
            ncol=len(handles) + len(family_handles), fontsize=6.8,
        )
        fig.suptitle(
            "Sensitivity concentration and selectivity across retrieval capacity",
            fontsize=12.2, y=0.985,
        )
        fig.subplots_adjust(
            left=0.06, right=0.995, bottom=0.24, top=0.79, wspace=0.36
        )
        _save_figure(
            fig,
            output_dir / "supplementary" / "capacity",
            ORGANISATION_FIGURE_STEM,
            {
                "paper_version": PAPER_VERSION,
                "point": "summaries are computed within each independently trained cell",
                "cross_N_head_alignment": False,
                "family_line_style": "solid semantic; dashed structural",
            },
        )
        plt.close(fig)


def family_overlap_rows(
    inputs: v2.PaperInputs,
    *,
    models: Sequence[str],
    seeds: Sequence[int],
    ns: Sequence[int],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for records in ns:
        for model in models:
            for seed in seeds:
                scores = inputs.score_bindings[
                    (str(model), int(records), int(seed))
                ].score_artifact.value
                role = inputs.role_results[(str(model), int(records), int(seed))]
                coordinates = scores["coordinates"]
                for core_name in v2.CORE_FAMILIES:
                    core = {tuple(map(int, item)) for item in scores["families"][core_name]}
                    for role_name in (
                        "address", "content", "address_control", "content_control"
                    ):
                        comparison = {
                            tuple(map(int, item))
                            for item in role["families"][role_name]
                        }
                        overlap = core & comparison
                        union = core | comparison
                        j_values = [
                            float(coordinates.joint_sensitivity[head])
                            for head in overlap
                        ]
                        d_values = [
                            float(coordinates.selectivity[head])
                            for head in overlap
                        ]
                        output.append(
                            {
                                "model": model,
                                "N": int(records),
                                "seed": int(seed),
                                "core_family": core_name,
                                "role_family": role_name,
                                "core_heads": len(core),
                                "role_heads": len(comparison),
                                "overlap_heads": len(overlap),
                                "jaccard": len(overlap) / len(union) if union else np.nan,
                                "core_recall": len(overlap) / len(core) if core else np.nan,
                                "role_precision": (
                                    len(overlap) / len(comparison)
                                    if comparison else np.nan
                                ),
                                "overlap_mean_J": (
                                    float(np.mean(j_values)) if j_values else np.nan
                                ),
                                "overlap_mean_D_rel": (
                                    float(np.mean(d_values)) if d_values else np.nan
                                ),
                                "overlap_layers": ",".join(
                                    map(str, sorted({head[0] + 1 for head in overlap}))
                                ),
                            }
                        )
    return output


def plot_family_overlap(
    rows: Sequence[Mapping[str, Any]],
    *,
    output_dir: Path,
    models: Sequence[str],
    ns: Sequence[int],
) -> None:
    import matplotlib.pyplot as plt

    pairings = (
        ("semantic_leaning", "address", "Semantic ↔ address"),
        ("semantic_leaning", "content", "Semantic ↔ content"),
        ("structural_leaning", "address", "Structural ↔ address"),
        ("structural_leaning", "content", "Structural ↔ content"),
    )
    with _style():
        fig, axes = plt.subplots(
            1, len(models), figsize=(3.15 * len(models), 2.9), squeeze=False
        )
        offsets = np.linspace(-0.24, 0.24, len(pairings))
        colours = (
            v2.SEMANTIC_COLOUR, "#D88B8B",
            v2.STRUCTURAL_COLOUR, "#82A9C3",
        )
        for column, model in enumerate(models):
            axis = axes[0, column]
            for offset, colour, (core, role, label) in zip(
                offsets, colours, pairings
            ):
                means = []
                for records in ns:
                    values = [
                        float(row["jaccard"]) for row in rows
                        if row["model"] == model
                        and int(row["N"]) == int(records)
                        and row["core_family"] == core
                        and row["role_family"] == role
                    ]
                    means.append(float(np.mean(values)) if values else np.nan)
                axis.plot(
                    np.asarray(ns) + offset, means,
                    color=colour, linewidth=1.1, marker="o", markersize=3.3,
                    label=label,
                )
            axis.set_xticks(ns)
            axis.set_ylim(-0.02, 1.02)
            axis.set_xlabel("Memory size, $N$")
            axis.set_title(MODEL_LABELS[str(model)], fontsize=9.5)
            if column == 0:
                axis.set_ylabel("Mean Jaccard overlap")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.005),
            ncol=4, fontsize=6.8,
        )
        fig.suptitle(
            "Task-defined role families within the canonical specialisation landscape",
            fontsize=11.8, y=0.985,
        )
        fig.subplots_adjust(
            left=0.08, right=0.995, bottom=0.25, top=0.79, wspace=0.28
        )
        _save_figure(
            fig,
            output_dir / "supplementary" / "roles",
            OVERLAP_FIGURE_STEM,
            {
                "paper_version": PAPER_VERSION,
                "interpretation": (
                    "descriptive bridge only; address/content families are independently frozen"
                ),
            },
        )
        plt.close(fig)


def discriminant_rows(
    causal_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int], dict[str, float]] = {}
    for row in causal_rows:
        if row["section"] != "raw_score_calibration":
            continue
        grouped.setdefault(
            (str(row["model"]), int(row["N"]), int(row["seed"])), {}
        )[str(row["test"])] = float(row["estimate"])
    output = []
    for (model, records, seed), values in sorted(grouped.items()):
        semantic = (
            values.get("S_semantic_vs_G_semantic", np.nan)
            - values.get("S_semantic_vs_G_structural_control", np.nan)
        )
        structural = (
            values.get("S_structural_vs_G_structural", np.nan)
            - values.get("S_structural_vs_G_semantic_control", np.nan)
        )
        output.extend(
            (
                {
                    "model": model, "N": records, "seed": seed,
                    "channel": "semantic", "specificity_gap": semantic,
                    "definition": "same-channel rho minus cross-channel rho",
                },
                {
                    "model": model, "N": records, "seed": seed,
                    "channel": "structural", "specificity_gap": structural,
                    "definition": "same-channel rho minus cross-channel rho",
                },
            )
        )
    return output


def semantic_validity_audit_rows(
    inputs: v2.PaperInputs,
    *,
    models: Sequence[str],
    seeds: Sequence[int],
    causal_ns: Sequence[int],
    causal_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Machine-readable controls for semantic/structural discriminant validity."""

    output: list[dict[str, Any]] = []
    for row in causal_rows:
        if row["section"] == "coordinate_validation":
            output.append(
                {
                    "model": row["model"],
                    "N": int(row["N"]),
                    "seed": int(row["seed"]),
                    "audit": "within_layer_permutation",
                    "target": row["test"],
                    "estimate": row["estimate"],
                    "control": row.get("p_within_layer", np.nan),
                    "difference": np.nan,
                    "interpretation": "control is within-layer permutation p-value",
                }
            )
    for records in causal_ns:
        for model in models:
            for seed in seeds:
                key = (str(model), int(records), int(seed))
                scores = inputs.score_bindings[key].score_artifact.value
                causal = inputs.causal[key]
                coordinates = scores["coordinates"]
                targets = causal.get("summary", {}).get("targets", {})
                clean = causal.get("clean_ablation", {})
                for family, channel in (
                    ("semantic_leaning", "semantic"),
                    ("structural_leaning", "structural"),
                ):
                    family_heads = tuple(scores["families"].get(family, ()))
                    control_name = f"{family}_central_control"
                    control_heads = tuple(
                        scores.get("matched_controls", {}).get(control_name, ())
                    )
                    family_j = np.asarray(
                        [
                            float(coordinates.joint_sensitivity[tuple(head)])
                            for head in family_heads
                        ],
                        dtype=np.float64,
                    )
                    control_j = np.asarray(
                        [
                            float(coordinates.joint_sensitivity[tuple(head)])
                            for head in control_heads
                        ],
                        dtype=np.float64,
                    )
                    output.append(
                        {
                            "model": model,
                            "N": int(records),
                            "seed": int(seed),
                            "audit": "matched_control_J",
                            "target": family,
                            "estimate": (
                                float(np.mean(family_j)) if len(family_j) else np.nan
                            ),
                            "control": (
                                float(np.mean(control_j)) if len(control_j) else np.nan
                            ),
                            "difference": (
                                float(np.mean(family_j) - np.mean(control_j))
                                if len(family_j) and len(control_j) else np.nan
                            ),
                            "interpretation": "families and controls should be comparable in J",
                        }
                    )
                    family_target = targets.get(f"family_{family}", {}).get(
                        "calibrated", {}
                    )
                    control_target = targets.get(
                        f"control_{control_name}", {}
                    ).get("calibrated", {})
                    for endpoint in ("g", "n"):
                        estimate = v2._as_float(
                            family_target.get(f"{endpoint}_{channel}")
                        )
                        control = v2._as_float(
                            control_target.get(f"{endpoint}_{channel}")
                        )
                        output.append(
                            {
                                "model": model,
                                "N": int(records),
                                "seed": int(seed),
                                "audit": "family_vs_matched_control",
                                "target": f"{family}:{endpoint}_{channel}",
                                "estimate": estimate,
                                "control": control,
                                "difference": estimate - control,
                                "interpretation": (
                                    "aligned calibrated causal response of frozen core family "
                                    "minus its independently matched central control"
                                ),
                            }
                        )
                    family_clean = clean.get(f"family_{family}", {})
                    control_clean = clean.get(f"control_{control_name}", {})
                    estimate = v2._as_float(
                        family_clean.get("prediction_movement")
                    )
                    control = v2._as_float(
                        control_clean.get("prediction_movement")
                    )
                    output.append(
                        {
                            "model": model,
                            "N": int(records),
                            "seed": int(seed),
                            "audit": "clean_ablation_family_vs_control",
                            "target": family,
                            "estimate": estimate,
                            "control": control,
                            "difference": estimate - control,
                            "interpretation": (
                                "clean prediction movement of frozen family minus matched control"
                            ),
                        }
                    )
    return output


def interaction_outlier_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Robust three-seed diagnostic for the demoted family-interaction endpoint."""

    grouped: dict[tuple[str, int, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        if row["section"] != "family_interaction":
            continue
        grouped.setdefault(
            (
                str(row["model"]),
                int(row["N"]),
                str(row["test"]),
                str(row["kind"]),
            ),
            [],
        ).append(row)
    output: list[dict[str, Any]] = []
    for (model, records, test, kind), selected in sorted(grouped.items()):
        values = np.asarray(
            [v2._as_float(row["estimate"]) for row in selected],
            dtype=np.float64,
        )
        finite = values[np.isfinite(values)]
        median = float(np.median(finite)) if len(finite) else np.nan
        mad = (
            float(np.median(np.abs(finite - median))) if len(finite) else np.nan
        )
        for row, value in zip(selected, values):
            robust_z = (
                float(0.6745 * (value - median) / mad)
                if np.isfinite(mad) and mad > 1e-12
                else np.nan
            )
            output.append(
                {
                    "model": model,
                    "N": records,
                    "seed": int(row["seed"]),
                    "test": test,
                    "kind": kind,
                    "estimate": value,
                    "seed_median": median,
                    "seed_MAD": mad,
                    "robust_z": robust_z,
                    "flag_abs_robust_z_gt_3_5": (
                        bool(abs(robust_z) > 3.5)
                        if np.isfinite(robust_z) else False
                    ),
                    "headline_status": "demoted_pending_outlier_audit",
                }
            )
    return output


def plot_discriminant_validity(
    rows: Sequence[Mapping[str, Any]],
    *,
    output_dir: Path,
    models: Sequence[str],
    seeds: Sequence[int],
    ns: Sequence[int],
) -> None:
    import matplotlib.pyplot as plt

    with _style():
        fig, axes = plt.subplots(
            1, len(models), figsize=(3.15 * len(models), 2.75), squeeze=False
        )
        for column, model in enumerate(models):
            axis = axes[0, column]
            for channel, colour in (
                ("semantic", v2.SEMANTIC_COLOUR),
                ("structural", v2.STRUCTURAL_COLOUR),
            ):
                matrix = []
                for seed in seeds:
                    values = []
                    for records in ns:
                        match = [
                            row for row in rows
                            if row["model"] == model
                            and int(row["N"]) == int(records)
                            and int(row["seed"]) == int(seed)
                            and row["channel"] == channel
                        ]
                        values.append(
                            float(match[0]["specificity_gap"]) if match else np.nan
                        )
                    matrix.append(values)
                    axis.plot(ns, values, color=colour, alpha=0.20, linewidth=0.65)
                axis.plot(
                    ns, np.nanmean(np.asarray(matrix), axis=0),
                    color=colour, linewidth=1.5, marker="o", markersize=3.5,
                    label=channel.capitalize(),
                )
            axis.axhline(0, color="#888888", linestyle="--", linewidth=0.7)
            axis.set_xticks(ns)
            axis.set_xlabel("Memory size, $N$")
            axis.set_title(MODEL_LABELS[str(model)], fontsize=9.5)
            if column == 0:
                axis.set_ylabel("Same − cross channel $\\rho$")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01),
            ncol=2, fontsize=7,
        )
        fig.suptitle(
            "Discriminant validity of canonical semantic and structural scores",
            fontsize=11.8, y=0.985,
        )
        fig.subplots_adjust(
            left=0.085, right=0.995, bottom=0.23, top=0.78, wspace=0.28
        )
        _save_figure(
            fig,
            output_dir / "supplementary" / "validity",
            DISCRIMINANT_FIGURE_STEM,
            {
                "paper_version": PAPER_VERSION,
                "positive": "same-channel association exceeds cross-channel association",
                "scope": "descriptive difference of held-out rank correlations",
            },
        )
        plt.close(fig)


def plot_complete_grounding(
    inputs: v2.PaperInputs,
    *,
    output_dir: Path,
    models: Sequence[str],
    performance_ns: Sequence[int],
    causal_ns: Sequence[int],
    seeds: Sequence[int],
    causal_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Accuracy plus complete J/D_rel causal heatmaps; no family-interaction panel."""

    import matplotlib.pyplot as plt

    j_mean, j_sd = v2._metric_matrix(
        causal_rows, models=models, ns=causal_ns,
        section="coordinate_validation",
        test="J_vs_clean_prediction_movement",
    )
    d_mean, d_sd = v2._metric_matrix(
        causal_rows, models=models, ns=causal_ns,
        section="coordinate_validation",
        test="D_rel_vs_gross_contrast",
    )
    with _style():
        fig, axes = plt.subplots(
            1, 3, figsize=(9.8, 3.35),
            gridspec_kw={"width_ratios": (1.55, 1, 1)},
        )
        for model in models:
            means, sds = [], []
            for records in performance_ns:
                mean, sd, _ = v2.accuracy_summary(
                    inputs.performance, str(model), int(records), seeds
                )
                means.append(mean)
                sds.append(sd)
            mean_array = np.asarray(means)
            sd_array = np.asarray(sds)
            axes[0].plot(
                performance_ns, mean_array,
                color=MODEL_COLOURS[str(model)],
                marker=MODEL_MARKERS[str(model)],
                linewidth=1.45, markersize=4.3,
                label=MODEL_LABELS[str(model)],
            )
            axes[0].fill_between(
                performance_ns,
                np.clip(mean_array - sd_array, 0, 1),
                np.clip(mean_array + sd_array, 0, 1),
                color=MODEL_COLOURS[str(model)], alpha=0.12, linewidth=0,
            )
        axes[0].set_xticks(performance_ns)
        axes[0].set_ylim(-0.02, 1.02)
        axes[0].set_xlabel("Memory size, $N$")
        axes[0].set_ylabel("Held-out accuracy")
        axes[0].set_title("A  Retrieval competence", fontsize=9.7)
        handles, labels = axes[0].get_legend_handles_labels()
        v2._annotated_matrix(
            fig, axes[1], j_mean, j_sd,
            models=models, ns=causal_ns,
            title=r"B  $J$ ↔ clean ablation",
            cmap="RdBu_r", vmin=-1, vmax=1,
        )
        v2._annotated_matrix(
            fig, axes[2], d_mean, d_sd,
            models=models, ns=causal_ns,
            title=r"C  $D_{rel}$ ↔ channel contrast",
            cmap="RdBu_r", vmin=-1, vmax=1,
        )
        fig.legend(
            handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.008),
            ncol=len(handles), fontsize=7.2,
        )
        fig.suptitle(
            "Retrieval competence and causal grounding across capacity regimes",
            fontsize=12.6, y=0.985,
        )
        fig.subplots_adjust(
            left=0.065, right=0.995, bottom=0.24, top=0.80, wspace=0.48
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
                "excluded": "unstable family × channel interaction panel",
            },
        )
        plt.close(fig)


def make_v3_figures(
    inputs: v2.PaperInputs,
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
    bootstrap: BootstrapPolicy,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Complete per-N landscapes.  The renderer is identical for every cell; only N=16 is promoted.
    for records in score_ns:
        v2.plot_core_specialisation(
            inputs,
            output_dir=output_dir,
            models=models,
            seeds=seeds,
            records=int(records),
            headline=True,
        )

    causal_rows = v2.causal_summary_rows(
        inputs, models=models, causal_ns=causal_ns, seeds=seeds
    )
    _write_csv(output_dir / "tables" / "core_causal_validation.csv", causal_rows)
    _write_csv(
        output_dir / "tables" / "family_interaction_diagnostic.csv",
        [row for row in causal_rows if row["section"] == "family_interaction"],
    )
    _write_csv(
        output_dir / "tables" / "family_interaction_outlier_audit.csv",
        interaction_outlier_rows(causal_rows),
    )
    plot_compact_causal_validation(
        inputs,
        output_dir=output_dir,
        models=models,
        seeds=seeds,
        records=int(headline_n),
        rows=causal_rows,
    )

    conditional = [
        row
        for records in score_ns
        for row in v2.conditional_family_rows(
            inputs, models=models, seeds=seeds, records=int(records)
        )
    ]
    _write_csv(
        output_dir / "tables" / "conditional_core_family_fingerprints.csv",
        conditional,
    )
    localisation = localisation_rows(conditional)
    _write_csv(output_dir / "tables" / "query_localisation.csv", localisation)

    counterfactual = v2.counterfactual_double_dissociation_rows(
        inputs.counterfactual,
        models=models,
        ns=counterfactual_ns,
        bootstrap=bootstrap,
    )
    counterfactual.extend(
        pooled_counterfactual_rows(
            inputs.counterfactual,
            models=models,
            ns=counterfactual_ns,
            bootstrap=bootstrap,
        )
    )
    _write_csv(
        output_dir / "tables" / "counterfactual_double_dissociation.csv",
        counterfactual,
    )
    plot_localisation_and_counterfactual(
        inputs,
        output_dir=output_dir,
        models=models,
        seeds=seeds,
        score_ns=score_ns,
        counterfactual_rows=counterfactual,
        localisation=localisation,
    )
    plot_conditional_fingerprint_supplement(
        inputs,
        output_dir=output_dir,
        models=models,
        seeds=seeds,
        records=int(headline_n),
        rows=conditional,
    )

    plot_complete_grounding(
        inputs,
        output_dir=output_dir,
        models=models,
        performance_ns=performance_ns,
        causal_ns=causal_ns,
        seeds=seeds,
        causal_rows=causal_rows,
    )

    organisation = organisation_rows(
        inputs, models=models, seeds=seeds, ns=score_ns
    )
    _write_csv(output_dir / "tables" / "head_organisation.csv", organisation)
    plot_organisation(
        organisation,
        output_dir=output_dir,
        models=models,
        seeds=seeds,
        ns=score_ns,
    )

    overlap = family_overlap_rows(
        inputs, models=models, seeds=seeds, ns=score_ns
    )
    _write_csv(output_dir / "tables" / "core_role_family_overlap.csv", overlap)
    plot_family_overlap(
        overlap, output_dir=output_dir, models=models, ns=score_ns
    )

    discriminant = discriminant_rows(causal_rows)
    _write_csv(
        output_dir / "tables" / "channel_discriminant_validity.csv",
        discriminant,
    )
    plot_discriminant_validity(
        discriminant,
        output_dir=output_dir,
        models=models,
        seeds=seeds,
        ns=causal_ns,
    )
    semantic_audit = semantic_validity_audit_rows(
        inputs,
        models=models,
        seeds=seeds,
        causal_ns=causal_ns,
        causal_rows=causal_rows,
    )
    _write_csv(
        output_dir / "tables" / "semantic_structural_validity_audit.csv",
        semantic_audit,
    )

    for records in carriage_ns:
        v2.plot_role_conditioned_carriage(
            inputs,
            output_dir=output_dir,
            models=models,
            seeds=seeds,
            records=int(records),
            bootstrap=bootstrap,
            headline=False,
        )

    atomic_json(
        output_dir / "paper_figure_index.json",
        {
            "paper_version": PAPER_VERSION,
            "headline_figures": [
                v2.CORE_FIGURE_STEM.format(N=int(headline_n)),
                CAUSAL_FIGURE_STEM.format(N=int(headline_n)),
                INTERPRETATION_FIGURE_STEM,
                TRANSITION_FIGURE_STEM,
            ],
            "complete_core_landscapes": [
                v2.CORE_FIGURE_STEM.format(N=int(records))
                for records in score_ns
            ],
            "supplementary_analyses": [
                FINGERPRINT_FIGURE_STEM.format(N=int(headline_n)),
                ORGANISATION_FIGURE_STEM,
                OVERLAP_FIGURE_STEM,
                DISCRIMINANT_FIGURE_STEM,
                "raw role-conditioned Functional carriage",
            ],
            "demoted_or_removed": {
                "family_interaction": "table only pending outlier audit",
                "R_role": "not used",
                "event_normalised_carriage": "not used",
                "head_intervals": "supplementary companions",
                "four_coordinate_role_spaghetti": "supplementary fingerprint only",
            },
            "figures": sorted(
                str(path.relative_to(output_dir))
                for path in output_dir.rglob("*.png")
            ),
            "tables": sorted(
                str(path.relative_to(output_dir))
                for path in output_dir.rglob("*.csv")
            ),
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cache-only v3 publication synthesis for the fixed-N NAR experiment"
    )
    parser.add_argument("--phase", choices=("figures",), default="figures")
    parser.add_argument(
        "--drive-root",
        default="/content/drive/MyDrive/graph_specialisation_metrics/nar_grit",
    )
    parser.add_argument("--training-run-name", default="nar_grit_fixed_n_v3")
    parser.add_argument("--base-analysis-name", default="canonical_nar_analysis_d128")
    parser.add_argument("--source-extension-name", default=SOURCE_EXTENSION_NAME)
    parser.add_argument(
        "--causal-extension-name", default=DEFAULT_CAUSAL_EXTENSION_NAME
    )
    parser.add_argument("--paper-analysis-name", default=DEFAULT_PAPER_ANALYSIS_NAME)
    parser.add_argument("--models", default="1hop,2hop,dense")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--cached-ns", default="4,16,64")
    parser.add_argument("--score-ns", default="4,8,16,32,64")
    parser.add_argument("--canonical-causal-ns", default="4,16,64")
    parser.add_argument("--transition-causal-ns", default="8,32")
    parser.add_argument("--counterfactual-ns", default="4,8,16")
    parser.add_argument("--carriage-ns", default="4,16,64")
    parser.add_argument("--performance-ns", default="4,8,16,32,64,80")
    parser.add_argument("--headline-n", type=int, default=16)
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
    canonical_causal_ns = _parse_csv_ints(args.canonical_causal_ns)
    transition_causal_ns = _parse_csv_ints(args.transition_causal_ns)
    causal_ns = tuple(
        value for value in score_ns
        if value in set(canonical_causal_ns) | set(transition_causal_ns)
    )
    counterfactual_ns = _parse_csv_ints(args.counterfactual_ns)
    carriage_ns = _parse_csv_ints(args.carriage_ns)
    performance_ns = _parse_csv_ints(args.performance_ns)
    if any(model not in MODEL_ORDER for model in models):
        raise ValueError(f"models must be drawn from {MODEL_ORDER}")
    if len(seeds) != 3:
        raise ValueError("the registered paper synthesis requires exactly three seeds")
    if not set(canonical_causal_ns).issubset(set(cached_ns)):
        raise ValueError("canonical-causal-ns must refer to canonical cached cells")
    if not set(transition_causal_ns).issubset(set(score_ns) - set(cached_ns)):
        raise ValueError(
            "transition-causal-ns must refer to non-canonical score-cached cells"
        )
    if set(causal_ns) != set(score_ns):
        raise ValueError("v3 requires causal validation for every score-ns cell")
    if int(args.headline_n) not in set(causal_ns):
        raise ValueError("headline-n must have scores and causal validation")
    if not set(carriage_ns).issubset(set(cached_ns)):
        raise ValueError("carriage-ns must refer to canonical cached cells")

    drive_root = Path(args.drive_root)
    training_run_dir = drive_root / str(args.training_run_name)
    base_analysis_root = training_run_dir / str(args.base_analysis_name)
    base_canonical_root = base_analysis_root / "canonical"
    source_extension_root = (
        base_analysis_root / "extensions" / str(args.source_extension_name)
    )
    causal_extension_root = (
        base_analysis_root / "extensions" / str(args.causal_extension_name)
    )
    output_dir = (
        base_analysis_root / "extensions" / str(args.paper_analysis_name)
    )
    _safe_extension_layout(base_analysis_root, output_dir)
    protected_sources = {
        source_extension_root.resolve(),
        causal_extension_root.resolve(),
        base_canonical_root.resolve(),
    }
    if output_dir.resolve() in protected_sources:
        raise ValueError("paper-analysis-name must select a new derived-output namespace")

    inputs, policies = load_v3_inputs(
        base_analysis_root=base_analysis_root,
        base_canonical_root=base_canonical_root,
        source_extension_root=source_extension_root,
        causal_extension_root=causal_extension_root,
        training_run_dir=training_run_dir,
        models=models,
        score_ns=score_ns,
        canonical_causal_ns=canonical_causal_ns,
        transition_causal_ns=transition_causal_ns,
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
    make_v3_figures(
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
        bootstrap=bootstrap,
    )
    atomic_json(
        output_dir / "run_summary.json",
        {
            "paper_version": PAPER_VERSION,
            "output_dir": str(output_dir),
            "base_canonical_root": str(base_canonical_root),
            "source_extension_root": str(source_extension_root),
            "causal_extension_root": str(causal_extension_root),
            "checkpoint_inference": False,
            "score_recomputation": False,
            "models": list(models),
            "seeds": list(seeds),
            "score_N_values": list(score_ns),
            "causal_N_values": list(causal_ns),
            "counterfactual_N_values": list(counterfactual_ns),
            "carriage_N_values": list(carriage_ns),
            "performance_N_values": list(performance_ns),
        },
    )
    print(f"[done] v3 paper figures and tables saved under {output_dir}", flush=True)
    return {"output_dir": str(output_dir)}


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    return run(build_parser().parse_args(list(argv) if argv is not None else None))


__all__ = [
    "DEFAULT_PAPER_ANALYSIS_NAME",
    "PAPER_VERSION",
    "build_parser",
    "discriminant_rows",
    "family_overlap_rows",
    "localisation_rows",
    "main",
    "organisation_rows",
    "pooled_counterfactual_rows",
    "run",
]


if __name__ == "__main__":
    main()
