"""Metric identifiability controls for local versus non-local structure.

This lightweight experiment asks what canonical semantic/structural head scores
and their carrier-distance breakdowns can identify after structural
preprocessing has already happened.

Two independent candidate values are presented and the target is the value on
the structurally selected candidate.  A one-parameter soft gate uses either a
partially reliable local clue or an exact multi-hop clue.  A third mechanism is
a behavioural twin of the exact-clue model whose semantic response is
registered at a distant carrier, providing a positive control for the distance
estimator.

The clean-output Jacobian and finite donor responses are combined exactly as in
the canonical projected-transport score: ``q = (z_clean-z_event) * dy/dz``.
Semantic and structural payload interventions affect disjoint captured heads.
Consequently, normalized selectivity records the same division of labour even
when clue quality and predictive performance differ.  Carrier distance remains
zero for locally consumed RRWP because the canonical source is the node holding
the already-computed PE payload, not the remote topology on which that payload
depends.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..methodology.scores import head_coordinates

plt.switch_backend("Agg")

BLUE = "#4477AA"
ORANGE = "#EE7733"
GREEN = "#228833"
PURPLE = "#AA3377"
RED = "#CC6677"
GREY = "#777777"
LIGHT_GREY = "#D7D7D7"
TEXT = "#222222"


@dataclass(frozen=True)
class Config:
    output_dir: Path
    train_examples: int = 8_192
    test_examples: int = 8_192
    seeds: tuple[int, ...] = tuple(range(8))
    short_clue_reliability: float = 0.75
    reliability_sweep: tuple[float, ...] = (0.55, 0.65, 0.75, 0.85, 0.95, 1.0)
    maximum_gate_mass: float = 0.995
    transport_distance: int = 4
    local_pe_origin_distance: int = 1
    multihop_pe_origin_distance: int = 4
    data_seed: int = 67_109

    def validate(self) -> None:
        if min(self.train_examples, self.test_examples) < 32:
            raise ValueError("train_examples and test_examples must be at least 32")
        if not self.seeds or len(self.seeds) != len(set(self.seeds)):
            raise ValueError("seeds must be non-empty and unique")
        if not 0.5 < self.short_clue_reliability < 1.0:
            raise ValueError("short_clue_reliability must lie in (0.5, 1)")
        if (
            not self.reliability_sweep
            or tuple(sorted(set(self.reliability_sweep))) != self.reliability_sweep
            or any(not 0.5 < value <= 1.0 for value in self.reliability_sweep)
        ):
            raise ValueError("reliability_sweep must be unique, increasing, and in (0.5,1]")
        if not 0.5 < self.maximum_gate_mass < 1.0:
            raise ValueError("maximum_gate_mass must lie in (0.5,1)")
        if min(
            self.transport_distance,
            self.local_pe_origin_distance,
            self.multihop_pe_origin_distance,
        ) < 0:
            raise ValueError("registered distances must be non-negative")


@dataclass(frozen=True)
class Split:
    values: np.ndarray
    target_index: np.ndarray

    @property
    def target(self) -> np.ndarray:
        return self.values[np.arange(len(self.values)), self.target_index]


@dataclass(frozen=True)
class Mechanism:
    name: str
    label: str
    reliability: float
    semantic_carrier_distance: int
    structural_carrier_distance: int
    pe_origin_distance: int


MECHANISM_ORDER = ("local_short", "local_multihop", "dense_transport")
MECHANISM_LABELS = {
    "local_short": "Local + partial short PE",
    "local_multihop": "Local + exact multi-hop PE",
    "dense_transport": "Exact PE + semantic transport",
}
MECHANISM_COLOURS = {
    "local_short": ORANGE,
    "local_multihop": BLUE,
    "dense_transport": PURPLE,
}
HEAD_NAMES = ("semantic_only", "semantic_leaning", "structural_leaning", "structural_only")
SEMANTIC_HEAD_LOADINGS = np.asarray((1.0, 0.75, 0.25, 0.0), dtype=np.float64)
STRUCTURAL_HEAD_LOADINGS = np.asarray((0.0, 0.25, 0.75, 1.0), dtype=np.float64)


def mechanisms(config: Config) -> tuple[Mechanism, ...]:
    return (
        Mechanism(
            name="local_short",
            label=MECHANISM_LABELS["local_short"],
            reliability=float(config.short_clue_reliability),
            semantic_carrier_distance=0,
            structural_carrier_distance=0,
            pe_origin_distance=int(config.local_pe_origin_distance),
        ),
        Mechanism(
            name="local_multihop",
            label=MECHANISM_LABELS["local_multihop"],
            reliability=1.0,
            semantic_carrier_distance=0,
            structural_carrier_distance=0,
            pe_origin_distance=int(config.multihop_pe_origin_distance),
        ),
        Mechanism(
            name="dense_transport",
            label=MECHANISM_LABELS["dense_transport"],
            reliability=1.0,
            semantic_carrier_distance=int(config.transport_distance),
            structural_carrier_distance=0,
            pe_origin_distance=int(config.multihop_pe_origin_distance),
        ),
    )


def generate_split(count: int, *, seed: int) -> Split:
    rng = np.random.default_rng(int(seed))
    return Split(
        values=rng.standard_normal((int(count), 2)),
        target_index=rng.integers(0, 2, size=int(count), dtype=np.int64),
    )


def generate_clue(
    target_index: np.ndarray,
    *,
    reliability: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    correct = rng.random(len(target_index)) < float(reliability)
    return np.where(correct, target_index, 1 - target_index).astype(np.int64)


def fit_gate(
    split: Split,
    clue_index: np.ndarray,
    *,
    maximum_gate_mass: float,
) -> dict[str, float]:
    """Fit the exact empirical MSE optimum for the one-parameter soft gate."""

    if clue_index.shape != split.target_index.shape:
        raise ValueError("clue_index does not align with the split")
    difference_squared = np.square(split.values[:, 0] - split.values[:, 1])
    correct = clue_index == split.target_index
    correct_mass = float(np.sum(difference_squared[correct]))
    total_mass = float(np.sum(difference_squared))
    if total_mass <= 0:
        raise ValueError("training values have no candidate contrast")
    unconstrained = correct_mass / total_mass
    lower = 1.0 - float(maximum_gate_mass)
    fitted_mass = float(np.clip(unconstrained, lower, float(maximum_gate_mass)))
    alpha = 0.5 * float(np.log(fitted_mass / (1.0 - fitted_mass)))
    return {
        "alpha": alpha,
        "fitted_clue_mass": fitted_mass,
        "unconstrained_clue_mass": unconstrained,
        "empirical_clue_accuracy": float(np.mean(correct)),
    }


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / np.sum(exponent, axis=-1, keepdims=True)


def evaluate_gate(
    split: Split,
    clue_index: np.ndarray,
    *,
    alpha: float,
    semantic_carrier_distance: int,
    structural_carrier_distance: int,
) -> tuple[dict[str, float], list[dict[str, float | int | str]]]:
    """Evaluate predictions and canonical projected-transport head scores."""

    candidate = np.arange(2, dtype=np.int64)[None, :]
    signs = np.where(candidate == clue_index[:, None], 1.0, -1.0)
    structural_state = float(alpha) * signs
    weights = _softmax(structural_state)
    prediction = np.sum(weights * split.values, axis=1)
    target = split.target
    clean_loss = np.square(prediction - target)

    semantic_event_mass: list[np.ndarray] = []
    structural_event_mass: list[np.ndarray] = []
    structural_output_movement: list[np.ndarray] = []
    structural_loss_increase: list[np.ndarray] = []
    for source in range(2):
        donor_value = np.roll(split.values[:, source], 1)
        semantic_delta = split.values[:, source] - donor_value
        semantic_gradient = weights[:, source]
        semantic_event_mass.append(np.abs(semantic_delta * semantic_gradient))

        structural_delta = structural_state[:, source] - structural_state[:, 1 - source]
        structural_gradient = weights[:, source] * (split.values[:, source] - prediction)
        structural_event_mass.append(np.abs(structural_delta * structural_gradient))

        event_state = np.array(structural_state, copy=True)
        event_state[:, source] = structural_state[:, 1 - source]
        event_prediction = np.sum(_softmax(event_state) * split.values, axis=1)
        structural_output_movement.append(np.abs(event_prediction - prediction))
        structural_loss_increase.append(
            np.square(event_prediction - target) - clean_loss
        )

    semantic_raw = float(np.mean(np.stack(semantic_event_mass)))
    structural_raw = float(np.mean(np.stack(structural_event_mass)))
    coordinates = head_coordinates(
        semantic_raw * SEMANTIC_HEAD_LOADINGS[None, :],
        structural_raw * STRUCTURAL_HEAD_LOADINGS[None, :],
        score_floor=1.0e-12,
        epsilon=1.0e-12,
        activity_floor=0.0,
    )
    if not coordinates.estimable:
        raise RuntimeError("registered score coordinates became non-estimable")

    correct_gate_mass = weights[np.arange(len(weights)), split.target_index]
    result = {
        "test_mse": float(np.mean(clean_loss)),
        "test_mae": float(np.mean(np.abs(prediction - target))),
        "mean_correct_candidate_mass": float(np.mean(correct_gate_mass)),
        "semantic_score_raw": semantic_raw,
        "structural_score_raw": structural_raw,
        "semantic_head_normalized_semantic": float(coordinates.normalized_semantic[0, 0]),
        "semantic_head_normalized_structural": float(
            coordinates.normalized_structural[0, 0]
        ),
        "semantic_head_J": float(coordinates.joint_sensitivity[0, 0]),
        "semantic_head_D_rel": float(coordinates.selectivity[0, 0]),
        "structural_head_normalized_semantic": float(
            coordinates.normalized_semantic[0, 3]
        ),
        "structural_head_normalized_structural": float(
            coordinates.normalized_structural[0, 3]
        ),
        "structural_head_J": float(coordinates.joint_sensitivity[0, 3]),
        "structural_head_D_rel": float(coordinates.selectivity[0, 3]),
        "structural_swap_output_movement": float(
            np.mean(np.stack(structural_output_movement))
        ),
        "structural_swap_loss_increase": float(
            np.mean(np.stack(structural_loss_increase))
        ),
    }
    for head, name in enumerate(HEAD_NAMES):
        result[f"{name}_J"] = float(coordinates.joint_sensitivity[0, head])
        result[f"{name}_D_rel"] = float(coordinates.selectivity[0, head])
    distance_rows: list[dict[str, float | int | str]] = []
    for channel, distance, mass, loadings in (
        (
            "semantic",
            semantic_carrier_distance,
            semantic_raw,
            SEMANTIC_HEAD_LOADINGS,
        ),
        (
            "structural",
            structural_carrier_distance,
            structural_raw,
            STRUCTURAL_HEAD_LOADINGS,
        ),
    ):
        total_loading = float(np.sum(loadings))
        for head, loading in zip(HEAD_NAMES, loadings, strict=True):
            if loading <= 0:
                continue
            distance_rows.append(
                {
                    "channel": channel,
                    "head": head,
                    "distance": int(distance),
                    "raw_mass": float(mass * loading),
                    "normalized_mass": float(loading / total_loading),
                }
            )
    return result, distance_rows


def run_mechanisms(config: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    distance_rows: list[dict[str, Any]] = []
    registered = mechanisms(config)
    for seed in config.seeds:
        train = generate_split(
            config.train_examples,
            seed=config.data_seed + int(seed),
        )
        test = generate_split(
            config.test_examples,
            seed=config.data_seed + 10_000 + int(seed),
        )
        clue_cache: dict[float, tuple[np.ndarray, np.ndarray, dict[str, float]]] = {}
        for reliability in sorted({item.reliability for item in registered}):
            train_clue = generate_clue(
                train.target_index,
                reliability=reliability,
                seed=config.data_seed + 20_000 + 101 * int(seed) + int(1_000 * reliability),
            )
            test_clue = generate_clue(
                test.target_index,
                reliability=reliability,
                seed=config.data_seed + 30_000 + 101 * int(seed) + int(1_000 * reliability),
            )
            fitted = fit_gate(
                train,
                train_clue,
                maximum_gate_mass=config.maximum_gate_mass,
            )
            clue_cache[float(reliability)] = (train_clue, test_clue, fitted)
        for mechanism in registered:
            _, test_clue, fitted = clue_cache[float(mechanism.reliability)]
            result, distances = evaluate_gate(
                test,
                test_clue,
                alpha=fitted["alpha"],
                semantic_carrier_distance=mechanism.semantic_carrier_distance,
                structural_carrier_distance=mechanism.structural_carrier_distance,
            )
            rows.append(
                {
                    "seed": int(seed),
                    "mechanism": mechanism.name,
                    "label": mechanism.label,
                    "clue_reliability": mechanism.reliability,
                    "semantic_carrier_distance": mechanism.semantic_carrier_distance,
                    "structural_carrier_distance": mechanism.structural_carrier_distance,
                    "pe_origin_distance": mechanism.pe_origin_distance,
                    **fitted,
                    **result,
                }
            )
            for distance in distances:
                distance_rows.append(
                    {
                        "seed": int(seed),
                        "mechanism": mechanism.name,
                        "label": mechanism.label,
                        "pe_origin_distance": mechanism.pe_origin_distance,
                        **distance,
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(distance_rows)


def run_reliability_sweep(config: Config) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seed in config.seeds:
        train = generate_split(config.train_examples, seed=config.data_seed + int(seed))
        test = generate_split(
            config.test_examples,
            seed=config.data_seed + 10_000 + int(seed),
        )
        for reliability in config.reliability_sweep:
            train_clue = generate_clue(
                train.target_index,
                reliability=reliability,
                seed=config.data_seed + 40_000 + 101 * int(seed) + int(1_000 * reliability),
            )
            test_clue = generate_clue(
                test.target_index,
                reliability=reliability,
                seed=config.data_seed + 50_000 + 101 * int(seed) + int(1_000 * reliability),
            )
            fitted = fit_gate(
                train,
                train_clue,
                maximum_gate_mass=config.maximum_gate_mass,
            )
            result, _ = evaluate_gate(
                test,
                test_clue,
                alpha=fitted["alpha"],
                semantic_carrier_distance=0,
                structural_carrier_distance=0,
            )
            rows.append(
                {
                    "seed": int(seed),
                    "clue_reliability": float(reliability),
                    **fitted,
                    **result,
                }
            )
    return pd.DataFrame(rows)


def _mean_ci(values: pd.Series) -> tuple[float, float]:
    array = values.to_numpy(dtype=np.float64)
    mean = float(np.mean(array))
    if array.size < 2:
        return mean, 0.0
    return mean, float(1.96 * np.std(array, ddof=1) / np.sqrt(array.size))


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "legend.fontsize": 8.2,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def _clean_axis(axis: Any) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(width=0.8)


def _panel_label(axis: Any, label: str) -> None:
    axis.text(
        -0.16,
        1.10,
        label,
        transform=axis.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        color=TEXT,
    )


def plot_mechanism_figure(
    results: pd.DataFrame,
    distances: pd.DataFrame,
    config: Config,
) -> tuple[Path, Path]:
    _configure_style()
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(13.3, 4.2),
        gridspec_kw={"width_ratios": (0.92, 1.05, 1.35)},
    )
    figure.subplots_adjust(left=0.06, right=0.985, bottom=0.20, top=0.75, wspace=0.35)
    figure.suptitle(
        "Specialisation detects channel organisation, not structural clue quality",
        fontsize=14.5,
        y=0.96,
        color=TEXT,
    )
    figure.text(
        0.5,
        0.865,
        "The exact multi-hop clue improves prediction without changing normalized head roles or canonical local response distance",
        ha="center",
        fontsize=9.3,
        color=GREY,
    )

    axis = axes[0]
    _panel_label(axis, "a")
    positions = np.arange(len(MECHANISM_ORDER))
    means, cis = [], []
    for mechanism in MECHANISM_ORDER:
        mean, ci = _mean_ci(results.loc[results["mechanism"] == mechanism, "test_mse"])
        means.append(mean)
        cis.append(ci)
    bars = axis.bar(
        positions,
        means,
        yerr=cis,
        capsize=3,
        color=[MECHANISM_COLOURS[name] for name in MECHANISM_ORDER],
        edgecolor="none",
        alpha=0.88,
    )
    for bar, mean in zip(bars, means, strict=True):
        label = f"{mean:.3f}" if mean >= 1.0e-3 else "<0.001"
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            mean + 0.018,
            label,
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=TEXT,
        )
    axis.set_title("Predictive behaviour differs")
    axis.set_ylabel("held-out MSE")
    axis.set_xticks(positions, ("short\nlocal", "multi-hop\nlocal", "semantic\ntransport"))
    axis.set_ylim(0.0, max(means) + 0.10)
    _clean_axis(axis)

    axis = axes[1]
    _panel_label(axis, "b")
    offsets = (-0.18, 0.0, 0.18)
    markers = ("o", "s", "D")
    for offset, marker, mechanism in zip(
        offsets, markers, MECHANISM_ORDER, strict=True
    ):
        subset = results[results["mechanism"] == mechanism]
        for head_index, column in enumerate(
            ("semantic_head_D_rel", "structural_head_D_rel")
        ):
            mean, ci = _mean_ci(subset[column])
            axis.errorbar(
                head_index + offset,
                mean,
                yerr=ci,
                marker=marker,
                markersize=6,
                color=MECHANISM_COLOURS[mechanism],
                linewidth=1.2,
                capsize=2,
                linestyle="none",
                label=MECHANISM_LABELS[mechanism] if head_index == 0 else None,
            )
    axis.axhline(0.0, color=LIGHT_GREY, linewidth=0.9)
    axis.set_title("Normalized head roles are identical (J = 1)")
    axis.set_ylabel(r"selectivity $D_{rel}$")
    axis.set_xticks((0, 1), ("semantic head", "structural head"))
    axis.set_xlim(-0.45, 1.45)
    axis.set_ylim(-1.12, 1.12)
    axis.legend(frameon=False, loc="center", bbox_to_anchor=(0.5, 0.51))
    _clean_axis(axis)

    axis = axes[2]
    _panel_label(axis, "c")
    for row, mechanism in enumerate(MECHANISM_ORDER[::-1]):
        subset = distances[distances["mechanism"] == mechanism]
        semantic_distance = float(
            subset.loc[subset["channel"] == "semantic", "distance"].mean()
        )
        structural_distance = float(
            subset.loc[subset["channel"] == "structural", "distance"].mean()
        )
        origin_distance = float(subset["pe_origin_distance"].mean())
        axis.scatter(
            semantic_distance,
            row + 0.11,
            s=58,
            marker="o",
            color=BLUE,
            label="semantic score mass" if row == 0 else None,
            zorder=3,
        )
        axis.scatter(
            structural_distance,
            row - 0.11,
            s=54,
            marker="s",
            color=GREEN,
            label="structural score mass" if row == 0 else None,
            zorder=3,
        )
        axis.scatter(
            origin_distance,
            row - 0.11,
            s=62,
            marker="x",
            linewidth=1.8,
            color=RED,
            label="PE dependency origin (audit)" if row == 0 else None,
            zorder=4,
        )
        axis.plot(
            (structural_distance, origin_distance),
            (row - 0.11, row - 0.11),
            color=RED,
            alpha=0.35,
            linewidth=1.0,
            zorder=1,
        )
    axis.set_title("Distance detects transport, not PE origin")
    axis.set_xlabel("distance from intervention source")
    axis.set_yticks(
        np.arange(3),
        tuple(MECHANISM_LABELS[name] for name in MECHANISM_ORDER[::-1]),
    )
    axis.set_xlim(-0.25, max(config.transport_distance, config.multihop_pe_origin_distance) + 0.35)
    axis.set_xticks(np.arange(0, max(config.transport_distance, config.multihop_pe_origin_distance) + 1))
    axis.set_ylim(-0.45, 2.45)
    axis.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, -0.22), ncol=2)
    _clean_axis(axis)

    figures = config.output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    png = figures / "03_specialisation_and_distance_identifiability.png"
    pdf = figures / "03_specialisation_and_distance_identifiability.pdf"
    figure.savefig(png, dpi=240, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return png, pdf


def plot_reliability_figure(sweep: pd.DataFrame, config: Config) -> tuple[Path, Path]:
    _configure_style()
    figure, axes = plt.subplots(1, 3, figsize=(13.1, 4.0))
    figure.subplots_adjust(left=0.065, right=0.985, bottom=0.17, top=0.75, wspace=0.34)
    figure.suptitle(
        "Clue quality changes usefulness while normalized specialisation stays fixed",
        fontsize=14.3,
        y=0.96,
        color=TEXT,
    )
    figure.text(
        0.5,
        0.865,
        "Raw projected response is non-monotonic; task loss supplies the missing alignment information",
        ha="center",
        fontsize=9.3,
        color=GREY,
    )
    reliability = np.asarray(config.reliability_sweep, dtype=np.float64)

    def series(column: str) -> tuple[np.ndarray, np.ndarray]:
        means, cis = [], []
        for value in reliability:
            mean, ci = _mean_ci(
                sweep.loc[np.isclose(sweep["clue_reliability"], value), column]
            )
            means.append(mean)
            cis.append(ci)
        return np.asarray(means), np.asarray(cis)

    axis = axes[0]
    _panel_label(axis, "a")
    for column, label, colour, marker in (
        ("test_mse", "clean test MSE", BLUE, "o"),
        (
            "structural_swap_loss_increase",
            "loss increase after structural swap",
            RED,
            "s",
        ),
    ):
        mean, ci = series(column)
        axis.plot(reliability, mean, color=colour, marker=marker, linewidth=1.9, label=label)
        axis.fill_between(reliability, mean - ci, mean + ci, color=colour, alpha=0.13)
    axis.set_title("Task-facing quantities track clue quality")
    axis.set_xlabel("structural clue reliability")
    axis.set_ylabel("loss")
    axis.set_xlim(0.53, 1.02)
    axis.set_ylim(bottom=-0.02)
    axis.legend(frameon=False, loc="center left")
    _clean_axis(axis)

    axis = axes[1]
    _panel_label(axis, "b")
    for column, label, colour, marker in (
        ("semantic_score_raw", r"raw $S_{sem}$", BLUE, "o"),
        ("structural_score_raw", r"raw $S_{str}$", GREEN, "s"),
    ):
        mean, ci = series(column)
        axis.plot(reliability, mean, color=colour, marker=marker, linewidth=1.9, label=label)
        axis.fill_between(reliability, mean - ci, mean + ci, color=colour, alpha=0.13)
    axis.set_title("Raw response is not a usefulness scale")
    axis.set_xlabel("structural clue reliability")
    axis.set_ylabel("projected transport mass")
    axis.set_xlim(0.53, 1.02)
    axis.set_ylim(bottom=-0.01)
    axis.legend(frameon=False, loc="center left")
    _clean_axis(axis)

    axis = axes[2]
    _panel_label(axis, "c")
    for column, label, colour, marker in (
        ("semantic_head_D_rel", "semantic head", BLUE, "o"),
        ("structural_head_D_rel", "structural head", GREEN, "s"),
    ):
        mean, ci = series(column)
        axis.plot(reliability, mean, color=colour, marker=marker, linewidth=1.9, label=label)
        axis.fill_between(reliability, mean - ci, mean + ci, color=colour, alpha=0.13)
    axis.axhline(0.0, color=LIGHT_GREY, linewidth=0.9)
    axis.set_title("Normalized roles are invariant (J = 1)")
    axis.set_xlabel("structural clue reliability")
    axis.set_ylabel(r"selectivity $D_{rel}$")
    axis.set_xlim(0.53, 1.02)
    axis.set_ylim(-1.12, 1.12)
    axis.legend(frameon=False, loc="center")
    _clean_axis(axis)

    figures = config.output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    png = figures / "04_structural_clue_reliability_sweep.png"
    pdf = figures / "04_structural_clue_reliability_sweep.pdf"
    figure.savefig(png, dpi=240, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return png, pdf


def _summary_by_mechanism(results: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for mechanism in MECHANISM_ORDER:
        subset = results[results["mechanism"] == mechanism]
        item: dict[str, float] = {}
        for column in (
            "test_mse",
            "semantic_score_raw",
            "structural_score_raw",
            "semantic_head_D_rel",
            "structural_head_D_rel",
            "semantic_head_J",
            "structural_head_J",
            "structural_swap_loss_increase",
        ):
            mean, ci = _mean_ci(subset[column])
            item[f"{column}_mean"] = mean
            item[f"{column}_95ci_half_width"] = ci
        item["semantic_carrier_distance"] = float(
            subset["semantic_carrier_distance"].iloc[0]
        )
        item["structural_carrier_distance"] = float(
            subset["structural_carrier_distance"].iloc[0]
        )
        item["pe_origin_distance"] = float(subset["pe_origin_distance"].iloc[0])
        summary[mechanism] = item
    return summary


def run(config: Config) -> dict[str, Any]:
    config.validate()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    results, distances = run_mechanisms(config)
    sweep = run_reliability_sweep(config)
    results.to_csv(config.output_dir / "mechanism_results.csv", index=False)
    distances.to_csv(config.output_dir / "distance_profiles.csv", index=False)
    sweep.to_csv(config.output_dir / "reliability_sweep.csv", index=False)
    main_figures = plot_mechanism_figure(results, distances, config)
    sweep_figures = plot_reliability_figure(sweep, config)

    local_multi = results[results["mechanism"] == "local_multihop"].set_index("seed")
    dense = results[results["mechanism"] == "dense_transport"].set_index("seed")
    behavioural_columns = (
        "test_mse",
        "semantic_score_raw",
        "structural_score_raw",
        "semantic_head_D_rel",
        "structural_head_D_rel",
    )
    result = {
        "config": {**asdict(config), "output_dir": str(config.output_dir)},
        "mechanisms": _summary_by_mechanism(results),
        "checks": {
            "local_multihop_and_dense_behavioural_twin": bool(
                all(
                    np.allclose(local_multi[column], dense[column], atol=1.0e-12, rtol=0.0)
                    for column in behavioural_columns
                )
            ),
            "normalized_coordinates_invariant": bool(
                results[
                    (
                        results["semantic_head_D_rel"]
                        - results["semantic_head_D_rel"].iloc[0]
                    ).abs()
                    < 1.0e-10
                ].shape[0]
                == len(results)
                and results[
                    (
                        results["structural_head_D_rel"]
                        - results["structural_head_D_rel"].iloc[0]
                    ).abs()
                    < 1.0e-10
                ].shape[0]
                == len(results)
            ),
            "four_head_mixed_landscape_invariant": bool(
                all(
                    np.allclose(
                        results[f"{head}_D_rel"],
                        results[f"{head}_D_rel"].iloc[0],
                        atol=1.0e-10,
                        rtol=0.0,
                    )
                    for head in HEAD_NAMES
                )
            ),
            "local_short_and_multihop_canonical_distances_equal": bool(
                results.loc[
                    results["mechanism"].isin(("local_short", "local_multihop")),
                    ["semantic_carrier_distance", "structural_carrier_distance"],
                ]
                .drop_duplicates()
                .shape[0]
                == 1
            ),
            "transport_positive_control_moves_semantic_distance": bool(
                int(dense["semantic_carrier_distance"].iloc[0])
                == int(config.transport_distance)
            ),
        },
        "figures": [str(path) for path in (*main_figures, *sweep_figures)],
    }
    with (config.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    return result


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(
        description="Run local/non-local specialisation identifiability controls."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/chapter6_local_nonlocal_specialisation_v1"),
    )
    parser.add_argument("--train-examples", type=int, default=8_192)
    parser.add_argument("--test-examples", type=int, default=8_192)
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--short-clue-reliability", type=float, default=0.75)
    parser.add_argument("--maximum-gate-mass", type=float, default=0.995)
    args = parser.parse_args(argv)
    config = Config(
        output_dir=args.output_dir,
        train_examples=int(args.train_examples),
        test_examples=int(args.test_examples),
        seeds=tuple(int(value) for value in args.seeds.split(",") if value),
        short_clue_reliability=float(args.short_clue_reliability),
        maximum_gate_mass=float(args.maximum_gate_mass),
    )
    result = run(config)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
