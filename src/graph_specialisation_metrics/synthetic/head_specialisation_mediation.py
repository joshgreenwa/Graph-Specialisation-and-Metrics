"""Controlled comparison of internal head specialisation and output mediation.

The graph support is a rooted path with carrier shells at distances 1--4. Six
hard-routed heads read either semantic node content or structural edge codes.
Within each channel, two heads read task-relevant signal (one nearby and one
redundant far copy), while a far head reads a strong in-distribution nuisance.
Only a minimum-norm scalar readout is fitted.

For channel ``x`` and head ``h``:

* ``S_x(h, d)`` is the finite internal head-response magnitude at carrier
  distance ``d``;
* ``M_x(h, d)`` is symmetric finite injection/restoration mediation of the
  scalar model output through the same head-distance cell.

The useful heads are positive controls. The output-silent specialists show why
internal selectivity can identify what a head processes without establishing
that the processed feature affects the final prediction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


PROTOCOL_VERSION = "head-specialisation-mediation-v1"
CHANNELS = ("semantic", "structural")
CHANNEL_COLOURS = {"semantic": "#0072B2", "structural": "#009E73"}
MEASURE_COLOURS = {"specialisation": "#6F6F6F", "mediation": "#CC79A7"}


@dataclass(frozen=True)
class Config:
    output_dir: Path
    train_samples: int = 4_096
    test_samples: int = 1_024
    seeds: tuple[int, ...] = (0, 1, 2, 3)
    decoy_scale: float = 2.0
    ridge: float = 1.0e-8
    bootstrap_replicates: int = 2_000
    analysis_seed: int = 83_117

    @property
    def fingerprint(self) -> str:
        payload = asdict(self)
        payload.pop("output_dir")
        payload["protocol_version"] = PROTOCOL_VERSION
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()[:20]

    def validate(self) -> None:
        if min(self.train_samples, self.test_samples) < 16:
            raise ValueError("train_samples and test_samples must be at least 16")
        if not self.seeds:
            raise ValueError("at least one seed is required")
        if self.decoy_scale <= 0 or self.ridge < 0:
            raise ValueError("decoy_scale must be positive and ridge non-negative")


@dataclass(frozen=True)
class HeadSpec:
    name: str
    short_name: str
    channel: str
    carrier_distance: int
    feature_index: int
    activation_scale: float
    role: str


@dataclass(frozen=True)
class Readout:
    weight: np.ndarray
    bias: float

    def predict(self, hidden: np.ndarray) -> np.ndarray:
        return hidden @ self.weight + float(self.bias)


def head_specs(decoy_scale: float) -> tuple[HeadSpec, ...]:
    """Registered hard-routing layout on a rooted path graph."""

    return (
        HeadSpec("semantic_local_signal", "Sem local", "semantic", 1, 0, 1.0, "signal"),
        HeadSpec(
            "semantic_far_redundant",
            "Sem far copy",
            "semantic",
            4,
            0,
            1.0,
            "redundant_signal",
        ),
        HeadSpec(
            "semantic_far_decoy",
            "Sem far decoy",
            "semantic",
            4,
            1,
            float(decoy_scale),
            "decoy",
        ),
        HeadSpec(
            "structural_mid_signal",
            "Str mid",
            "structural",
            2,
            2,
            1.0,
            "signal",
        ),
        HeadSpec(
            "structural_far_redundant",
            "Str far copy",
            "structural",
            4,
            2,
            1.0,
            "redundant_signal",
        ),
        HeadSpec(
            "structural_far_decoy",
            "Str far decoy",
            "structural",
            4,
            3,
            float(decoy_scale),
            "decoy",
        ),
    )


def generate_inputs(count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate in-distribution semantic and structural graph factors.

    Columns are semantic signal, semantic nuisance, structural edge signal,
    and structural edge nuisance. Structural signs index which of two legal
    leaf attachments is present at the registered carrier shell.
    """

    rng = np.random.default_rng(int(seed))
    factors = rng.choice(np.asarray([-1.0, 1.0]), size=(int(count), 4), replace=True)
    target = factors[:, 0] + factors[:, 2]
    return factors.astype(np.float64), target.astype(np.float64)


def head_activations(factors: np.ndarray, specs: Sequence[HeadSpec]) -> np.ndarray:
    return np.stack(
        [
            float(spec.activation_scale) * factors[:, int(spec.feature_index)]
            for spec in specs
        ],
        axis=1,
    )


def fit_readout(hidden: np.ndarray, target: np.ndarray, ridge: float) -> Readout:
    """Fit the symmetric minimum-norm readout over redundant head features."""

    hidden_mean = np.mean(hidden, axis=0)
    target_mean = float(np.mean(target))
    centred_hidden = hidden - hidden_mean
    centred_target = target - target_mean
    left, singular, right = np.linalg.svd(centred_hidden, full_matrices=False)
    multiplier = singular / (singular**2 + float(ridge))
    weight = right.T @ (multiplier * (left.T @ centred_target))
    bias = target_mean - float(hidden_mean @ weight)
    return Readout(weight=weight, bias=bias)


def channel_donor(factors: np.ndarray, channel: str) -> np.ndarray:
    """Flip both signal and nuisance within a legal independent channel."""

    donor = np.array(factors, copy=True)
    columns = (0, 1) if channel == "semantic" else (2, 3)
    donor[:, columns] *= -1.0
    return donor


def measure_seed(
    config: Config,
    *,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    specs = head_specs(config.decoy_scale)
    train_x, train_y = generate_inputs(config.train_samples, seed=config.analysis_seed + seed)
    test_x, test_y = generate_inputs(
        config.test_samples,
        seed=config.analysis_seed + 10_000 + seed,
    )
    train_hidden = head_activations(train_x, specs)
    test_hidden = head_activations(test_x, specs)
    readout = fit_readout(train_hidden, train_y, config.ridge)
    prediction = readout.predict(test_hidden)
    rows: list[dict[str, Any]] = []
    for channel in CHANNELS:
        donor_x = channel_donor(test_x, channel)
        donor_hidden = head_activations(donor_x, specs)
        clean_output = readout.predict(test_hidden)
        donor_output = readout.predict(donor_hidden)
        for head, spec in enumerate(specs):
            internal_delta = donor_hidden[:, head] - test_hidden[:, head]
            internal_score = float(np.mean(np.abs(internal_delta)))

            injected = np.array(test_hidden, copy=True)
            injected[:, head] = donor_hidden[:, head]
            restored = np.array(donor_hidden, copy=True)
            restored[:, head] = test_hidden[:, head]
            injection_effect = readout.predict(injected) - clean_output
            restoration_effect = donor_output - readout.predict(restored)
            mediation = float(
                0.5
                * np.mean(np.abs(injection_effect) + np.abs(restoration_effect))
            )
            rows.append(
                {
                    "seed": int(seed),
                    "head": spec.name,
                    "short_name": spec.short_name,
                    "head_channel": spec.channel,
                    "donor_channel": channel,
                    "distance": int(spec.carrier_distance),
                    "role": spec.role,
                    "readout_weight": float(readout.weight[head]),
                    "S_raw": internal_score,
                    "M_raw": mediation,
                }
            )
    for channel in CHANNELS:
        selected = [row for row in rows if row["donor_channel"] == channel]
        s_total = sum(float(row["S_raw"]) for row in selected)
        m_total = sum(float(row["M_raw"]) for row in selected)
        for row in selected:
            row["S_norm"] = float(row["S_raw"]) / max(s_total, 1.0e-15)
            row["M_norm"] = float(row["M_raw"]) / max(m_total, 1.0e-15)
    return rows, {
        "seed": int(seed),
        "train_samples": int(config.train_samples),
        "test_samples": int(config.test_samples),
        "test_mae": float(np.mean(np.abs(prediction - test_y))),
        "readout_weights": json.dumps(readout.weight.tolist()),
        "readout_bias": float(readout.bias),
    }


def _bootstrap(
    values: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values))
    if values.size < 2 or int(replicates) < 2:
        return mean, mean, mean
    rng = np.random.default_rng(int(seed))
    samples = np.empty(int(replicates), dtype=np.float64)
    for index in range(int(replicates)):
        selected = rng.integers(0, values.size, values.size)
        samples[index] = float(np.mean(values[selected]))
    return mean, float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def summarise(
    config: Config,
    rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    head_summary: list[dict[str, Any]] = []
    cells = sorted({(row["head"], row["donor_channel"]) for row in rows})
    for cell_index, (head, channel) in enumerate(cells):
        selected = [
            row
            for row in rows
            if row["head"] == head and row["donor_channel"] == channel
        ]
        template = selected[0]
        record: dict[str, Any] = {
            key: template[key]
            for key in ("head", "short_name", "head_channel", "donor_channel", "distance", "role")
        }
        record["readout_weight"] = float(
            np.mean([float(row["readout_weight"]) for row in selected])
        )
        for metric in ("S_raw", "M_raw", "S_norm", "M_norm"):
            mean, low, high = _bootstrap(
                np.asarray([float(row[metric]) for row in selected]),
                replicates=config.bootstrap_replicates,
                seed=config.analysis_seed + 97 * cell_index + len(metric),
            )
            record[f"{metric}_mean"] = mean
            record[f"{metric}_low"] = low
            record[f"{metric}_high"] = high
        head_summary.append(record)

    distance_summary: list[dict[str, Any]] = []
    for channel in CHANNELS:
        for metric in ("S_norm", "M_norm"):
            for distance in range(5):
                per_seed: list[float] = []
                for seed in config.seeds:
                    per_seed.append(
                        sum(
                            float(row[metric])
                            for row in rows
                            if row["donor_channel"] == channel
                            and int(row["seed"]) == int(seed)
                            and int(row["distance"]) == distance
                        )
                    )
                mean, low, high = _bootstrap(
                    np.asarray(per_seed),
                    replicates=config.bootstrap_replicates,
                    seed=config.analysis_seed + 211 * distance + len(metric) + len(channel),
                )
                distance_summary.append(
                    {
                        "channel": channel,
                        "metric": metric,
                        "distance": distance,
                        "mean": mean,
                        "low": low,
                        "high": high,
                    }
                )

    channel_summary: list[dict[str, Any]] = []
    for channel_index, channel in enumerate(CHANNELS):
        s_expected: list[float] = []
        m_expected: list[float] = []
        s_decoy: list[float] = []
        m_decoy: list[float] = []
        for seed in config.seeds:
            selected = [
                row
                for row in rows
                if row["donor_channel"] == channel and int(row["seed"]) == int(seed)
            ]
            s_expected.append(
                sum(float(row["distance"]) * float(row["S_norm"]) for row in selected)
            )
            m_expected.append(
                sum(float(row["distance"]) * float(row["M_norm"]) for row in selected)
            )
            s_decoy.append(
                sum(float(row["S_norm"]) for row in selected if row["role"] == "decoy")
            )
            m_decoy.append(
                sum(float(row["M_norm"]) for row in selected if row["role"] == "decoy")
            )
        record: dict[str, Any] = {"channel": channel}
        for metric_index, (metric, values) in enumerate(
            (
                ("S_expected_distance", s_expected),
                ("M_expected_distance", m_expected),
                ("S_decoy_share", s_decoy),
                ("M_decoy_share", m_decoy),
            )
        ):
            mean, low, high = _bootstrap(
                np.asarray(values),
                replicates=config.bootstrap_replicates,
                seed=config.analysis_seed + 307 * channel_index + metric_index,
            )
            record[f"{metric}_mean"] = mean
            record[f"{metric}_low"] = low
            record[f"{metric}_high"] = high
        channel_summary.append(record)
    return head_summary, distance_summary, channel_summary


def plot_headline(
    config: Config,
    head_summary: Sequence[dict[str, Any]],
    distance_summary: Sequence[dict[str, Any]],
    seed_summary: Sequence[dict[str, Any]],
) -> dict[str, str]:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.25))
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.17, top=0.72, wspace=0.34)

    order = [spec.name for spec in head_specs(config.decoy_scale)]
    active = {
        str(row["head"]): row
        for row in head_summary
        if row["head_channel"] == row["donor_channel"]
    }
    for position, head in enumerate(order):
        row = active[head]
        s_value = float(row["S_norm_mean"])
        m_value = float(row["M_norm_mean"])
        axes[0].plot(
            [s_value, m_value],
            [position, position],
            color="#BBBBBB",
            linewidth=1.5,
            zorder=1,
        )
        axes[0].errorbar(
            s_value,
            position,
            xerr=np.asarray(
                [
                    [s_value - float(row["S_norm_low"])],
                    [float(row["S_norm_high"]) - s_value],
                ]
            ),
            color=MEASURE_COLOURS["specialisation"],
            marker="^",
            markersize=7,
            linestyle="none",
            label="Internal specialisation" if position == 0 else None,
            zorder=2,
        )
        axes[0].errorbar(
            m_value,
            position,
            xerr=np.asarray(
                [
                    [m_value - float(row["M_norm_low"])],
                    [float(row["M_norm_high"]) - m_value],
                ]
            ),
            color=MEASURE_COLOURS["mediation"],
            marker="o",
            markersize=6,
            linestyle="none",
            label="Output mediation" if position == 0 else None,
            zorder=3,
        )
    axes[0].axvline(0, color="#AAAAAA", linewidth=1)
    axes[0].set_xlim(-0.02, 0.55)
    axes[0].set_yticks(
        np.arange(len(order)),
        [str(active[head]["short_name"]) for head in order],
    )
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Normalised channel mass")
    axes[0].set_title("Per-head specialisation and mediation")

    for axis, channel in zip(axes[1:], CHANNELS, strict=True):
        for metric, label, linestyle, marker in (
            ("S_norm", "Internal specialisation", "--", "^"),
            ("M_norm", "Output mediation", "-", "o"),
        ):
            selected = [
                row
                for row in distance_summary
                if row["channel"] == channel and row["metric"] == metric
            ]
            selected.sort(key=lambda row: int(row["distance"]))
            x = np.asarray([int(row["distance"]) for row in selected])
            y = np.asarray([float(row["mean"]) for row in selected])
            low = np.asarray([float(row["low"]) for row in selected])
            high = np.asarray([float(row["high"]) for row in selected])
            axis.plot(
                x,
                y,
                color=MEASURE_COLOURS[
                    "specialisation" if metric == "S_norm" else "mediation"
                ],
                linestyle=linestyle,
                marker=marker,
                linewidth=2,
                label=label,
            )
            axis.fill_between(
                x,
                low,
                high,
                color=MEASURE_COLOURS[
                    "specialisation" if metric == "S_norm" else "mediation"
                ],
                alpha=0.12,
            )
        axis.set_xlim(-0.15, 4.15)
        axis.set_ylim(-0.02, 0.8)
        axis.set_xticks(range(5))
        axis.set_xlabel("Carrier distance")
        axis.set_ylabel("Normalised channel mass")
        axis.set_title(f"{channel.title()} distance profile")
    axes[1].legend(frameon=False, fontsize=8, loc="upper left")

    mae = float(np.mean([float(row["test_mae"]) for row in seed_summary]))
    fig.suptitle("Internal head specialisation versus finite output mediation", fontsize=15, y=0.97)
    fig.text(
        0.5,
        0.88,
        (
            "Hard-routed graph heads with learned minimum-norm readout; "
            f"mean test MAE={mae:.2e}"
        ),
        ha="center",
        color="#666666",
        fontsize=9,
    )
    figure_dir = config.output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    png = figure_dir / "head_specialisation_vs_mediation.png"
    pdf = figure_dir / "head_specialisation_vs_mediation.pdf"
    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png), "pdf": str(pdf)}


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def run(config: Config) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    seed_summary: list[dict[str, Any]] = []
    for seed in config.seeds:
        seed_rows, health = measure_seed(config, seed=int(seed))
        rows.extend(seed_rows)
        seed_summary.append(health)
        print(f"[seed {seed}] test MAE={float(health['test_mae']):.3e}")
    head_summary, distance_summary, channel_summary = summarise(config, rows)
    result_dir = config.output_dir / "results"
    _write_csv(result_dir / "head_channel_rows.csv", rows)
    _write_csv(result_dir / "head_summary.csv", head_summary)
    _write_csv(result_dir / "distance_summary.csv", distance_summary)
    _write_csv(result_dir / "channel_summary.csv", channel_summary)
    _write_csv(result_dir / "seed_summary.csv", seed_summary)
    figures = plot_headline(config, head_summary, distance_summary, seed_summary)
    _write_json(
        result_dir / "summary.json",
        {
            "protocol_version": PROTOCOL_VERSION,
            "fingerprint": config.fingerprint,
            "figures": figures,
            "channel_summary": channel_summary,
            "interpretation": (
                "S_x measures finite internal selectivity; M_x measures the portion of the "
                "finite output response mediated by the same head-distance cell."
            ),
        },
    )
    return {
        "config": config,
        "head_summary": head_summary,
        "distance_summary": distance_summary,
        "channel_summary": channel_summary,
        "seed_summary": seed_summary,
        "figures": figures,
    }


def _parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/head_specialisation_mediation_v1"),
    )
    parser.add_argument("--train-samples", type=int, default=4_096)
    parser.add_argument("--test-samples", type=int, default=1_024)
    parser.add_argument("--seeds", default="0,1,2,3")
    parser.add_argument("--decoy-scale", type=float, default=2.0)
    parser.add_argument("--ridge", type=float, default=1.0e-8)
    parser.add_argument("--bootstrap-replicates", type=int, default=2_000)
    parser.add_argument("--analysis-seed", type=int, default=83_117)
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    config = Config(
        output_dir=args.output_dir,
        train_samples=int(args.train_samples),
        test_samples=int(args.test_samples),
        seeds=_parse_ints(args.seeds),
        decoy_scale=float(args.decoy_scale),
        ridge=float(args.ridge),
        bootstrap_replicates=int(args.bootstrap_replicates),
        analysis_seed=int(args.analysis_seed),
    )
    config.validate()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        config.output_dir / "config.json",
        {
            **asdict(config),
            "output_dir": str(config.output_dir),
            "fingerprint": config.fingerprint,
            "protocol_version": PROTOCOL_VERSION,
        },
    )
    result = run(config)
    print(f"[figure] {result['figures']['png']}")
    return result


if __name__ == "__main__":  # pragma: no cover
    main()
