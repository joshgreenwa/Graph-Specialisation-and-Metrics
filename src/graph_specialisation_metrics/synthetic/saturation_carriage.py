"""Four-seed finite-intervention range experiment under nonlinear saturation.

The experiment contains one source, a near carrier at distance one, and a far carrier
at distance ``D``:

    h_near = -gamma * b
    h_far  = (1 + gamma) * tanh(kappa * b) / tanh(kappa)
    y_hat  = h_near + h_far

The binary evaluation inputs are ``b in {-1, +1}``. A semantic donor replacement is
therefore the unique non-trivial intervention ``b -> -b``. Saturation changes the local
Jacobian of the far pathway without changing either endpoint of the finite intervention.

The two pathway coefficients are learned from a dense continuous teacher grid. Training
is intentionally tiny: there are only two scalar parameters. Checkpoints and measurements
are fingerprinted and saved independently so publication figures can be regenerated
without training or model inference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from ..methodology.carriage import beneficial_carriage, functional_carriage


PROTOCOL_VERSION = "saturation-carriage-v1"
DEFAULT_SEEDS = (0, 1, 2, 3)
DEFAULT_KAPPAS = (0.5, 1.0, 2.0, 4.0, 8.0)
DTYPE = torch.float64


@dataclass(frozen=True)
class ExperimentConfig:
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    kappas: tuple[float, ...] = DEFAULT_KAPPAS
    distance: int = 8
    gamma: float = 1.0
    target_alignment: float = 1.0
    train_points: int = 513
    train_steps: int = 600
    learning_rate: float = 0.05
    early_stop: float = 1.0e-13
    integrated_atol: float = 1.0e-10
    integrated_rtol: float = 1.0e-9
    integrated_max_intervals: int = 128
    integrated_tolerance: float = 1.0e-8

    def __post_init__(self) -> None:
        if len(self.seeds) != len(set(self.seeds)) or not self.seeds:
            raise ValueError("seeds must be non-empty and unique")
        if len(self.kappas) != len(set(self.kappas)) or not self.kappas:
            raise ValueError("kappas must be non-empty and unique")
        if any(value <= 0 for value in self.kappas):
            raise ValueError("every kappa must be positive")
        if self.distance <= 1:
            raise ValueError("distance must exceed one")
        if self.gamma <= 0:
            raise ValueError("gamma must be positive")
        if self.train_points < 3 or self.train_steps < 1:
            raise ValueError("training grid and step count must be positive")

    @property
    def record(self) -> dict[str, Any]:
        return {
            "protocol_version": PROTOCOL_VERSION,
            **asdict(self),
            "seeds": list(self.seeds),
            "kappas": list(self.kappas),
        }

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(self.record, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


def phi(value: torch.Tensor, kappa: float) -> torch.Tensor:
    """Smooth saturating map with exact endpoints ``phi(+/-1)=+/-1``."""

    return torch.tanh(float(kappa) * value) / math.tanh(float(kappa))


class SaturatedTwoPathStudent(nn.Module):
    """Known nonlinear basis with two learned pathway coefficients."""

    def __init__(self, *, kappa: float, init_seed: int) -> None:
        super().__init__()
        self.kappa = float(kappa)
        generator = torch.Generator(device="cpu").manual_seed(int(init_seed))
        self.near_weight = nn.Parameter(
            0.5 * torch.randn((), generator=generator, dtype=DTYPE)
        )
        self.far_weight = nn.Parameter(
            0.5 * torch.randn((), generator=generator, dtype=DTYPE)
        )

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        """Return ``[..., carrier=2, width=1]`` ordered as near, far."""

        source = source.to(dtype=DTYPE)
        near = self.near_weight * source
        far = self.far_weight * phi(source, self.kappa)
        return torch.stack((near, far), dim=-1).unsqueeze(-1)


def teacher_states(source: torch.Tensor, *, gamma: float, kappa: float) -> torch.Tensor:
    near = -float(gamma) * source
    far = (1.0 + float(gamma)) * phi(source, kappa)
    return torch.stack((near, far), dim=-1).unsqueeze(-1)


def _kappa_slug(kappa: float) -> str:
    return f"{float(kappa):g}".replace("-", "m").replace(".", "p")


def _checkpoint_path(output_dir: Path, seed: int, kappa: float) -> Path:
    return (
        output_dir
        / "cache"
        / "checkpoints"
        / f"seed_{int(seed):03d}"
        / f"kappa_{_kappa_slug(kappa)}.pt"
    )


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(temporary, path)


def _atomic_torch(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _load_torch(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch before the weights_only keyword.
        return torch.load(path, map_location="cpu")


def ensure_contract(output_dir: Path, config: ExperimentConfig) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "experiment.json"
    expected = {"fingerprint": config.fingerprint, "config": config.record}
    if path.exists():
        observed = json.loads(path.read_text())
        if observed != expected:
            raise RuntimeError(
                f"{path} belongs to a different experiment contract. "
                "Choose a new --output-dir rather than mixing caches."
            )
    else:
        _atomic_json(path, expected)


def train_one(config: ExperimentConfig, *, seed: int, kappa: float) -> dict[str, Any]:
    """Fit the two pathway coefficients to the known continuous teacher."""

    init_seed = 100_003 * int(seed) + int(round(1_000 * float(kappa))) + 17
    model = SaturatedTwoPathStudent(kappa=kappa, init_seed=init_seed)
    source = torch.linspace(-1.0, 1.0, config.train_points, dtype=DTYPE)
    target = teacher_states(source, gamma=config.gamma, kappa=kappa)
    optimiser = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history: list[dict[str, float | int]] = []
    started = time.time()

    for step in range(config.train_steps):
        prediction = model(source)
        loss = torch.nn.functional.mse_loss(prediction, target)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        if step == 0 or (step + 1) % max(1, config.train_steps // 20) == 0:
            history.append({"step": step + 1, "loss": float(loss.detach())})
        if step >= 20 and float(loss.detach()) <= config.early_stop:
            break

    # The objective is a two-variable convex least-squares problem. A final closed-form
    # polish removes optimiser residue at b=+/-1; otherwise a 1e-7 endpoint overshoot can
    # place the MAE kink microscopically close to the path endpoint and make an exact,
    # otherwise trivial line integral unnecessarily difficult to resolve numerically.
    with torch.no_grad():
        near_basis = source
        far_basis = phi(source, kappa)
        model.near_weight.copy_(
            torch.sum(near_basis * target[:, 0, 0]) / torch.sum(near_basis.square())
        )
        model.far_weight.copy_(
            torch.sum(far_basis * target[:, 1, 0]) / torch.sum(far_basis.square())
        )

    with torch.no_grad():
        final_loss = float(torch.nn.functional.mse_loss(model(source), target))
        endpoint = model(torch.tensor([-1.0, 1.0], dtype=DTYPE))
        endpoint_target = teacher_states(
            torch.tensor([-1.0, 1.0], dtype=DTYPE),
            gamma=config.gamma,
            kappa=kappa,
        )
        endpoint_error = float((endpoint - endpoint_target).abs().max())
    return {
        "protocol_version": PROTOCOL_VERSION,
        "fingerprint": config.fingerprint,
        "seed": int(seed),
        "kappa": float(kappa),
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "training": {
            "steps": int(step + 1),
            "final_loss": final_loss,
            "endpoint_error": endpoint_error,
            "seconds": round(time.time() - started, 4),
            "history": history,
            "least_squares_polish": True,
        },
    }


def ensure_checkpoints(
    output_dir: Path,
    config: ExperimentConfig,
    *,
    progress: bool = True,
) -> list[Path]:
    saved: list[Path] = []
    for seed in config.seeds:
        for kappa in config.kappas:
            path = _checkpoint_path(output_dir, seed, kappa)
            if path.exists():
                payload = _load_torch(path)
                if payload.get("fingerprint") != config.fingerprint:
                    raise RuntimeError(f"checkpoint contract mismatch: {path}")
                action = "reuse"
            else:
                payload = train_one(config, seed=seed, kappa=kappa)
                _atomic_torch(path, payload)
                action = "train"
            saved.append(path)
            if progress:
                training = payload["training"]
                print(
                    f"[{action}] seed={seed} kappa={kappa:g} "
                    f"loss={training['final_loss']:.2e} "
                    f"endpoint={training['endpoint_error']:.2e}",
                    flush=True,
                )
    return saved


def _model_from_checkpoint(path: Path, config: ExperimentConfig) -> tuple[nn.Module, dict]:
    payload = _load_torch(path)
    if payload.get("fingerprint") != config.fingerprint:
        raise RuntimeError(f"checkpoint contract mismatch: {path}")
    model = SaturatedTwoPathStudent(
        kappa=float(payload["kappa"]),
        init_seed=0,
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def _weighted_distance(values: np.ndarray, distances: np.ndarray) -> float:
    mass = float(values.sum())
    if not np.isfinite(mass) or mass <= 0:
        return float("nan")
    return float(np.sum(values * distances) / mass)


def measure_model(
    model: SaturatedTwoPathStudent,
    config: ExperimentConfig,
) -> dict[str, float]:
    """Measure the trained map at both binary source values."""

    distance = np.asarray([1.0, float(config.distance)], dtype=np.float64)
    rows: list[dict[str, float]] = []
    for source_value in (-1.0, 1.0):
        source = torch.tensor([source_value], dtype=DTYPE, requires_grad=True)
        event_source = torch.tensor([-source_value], dtype=DTYPE)
        clean = model(source).squeeze(0)
        event = model(event_source).squeeze(0)

        jacobian = torch.autograd.functional.jacobian(
            lambda value: model(value).squeeze(0).squeeze(-1),
            source,
            vectorize=True,
        ).reshape(2)
        jacobian_mass = jacobian.abs().detach().cpu().numpy()

        delta = (clean - event).reshape(1, 1, 2, 1)
        output_gradient = torch.ones((1, 2, 1), dtype=DTYPE)
        functional = (
            functional_carriage(delta, output_gradient)[:, 0].detach().cpu().numpy()
        )

        target = float(config.target_alignment) * source_value

        def loss_from_pooled(pooled: torch.Tensor) -> torch.Tensor:
            return (pooled[:, 0] - target).abs()

        integrated = beneficial_carriage(
            clean.detach(),
            event.detach().reshape(1, 1, 2, 1),
            loss_from_pooled,
            pooling="add",
            atol=config.integrated_atol,
            rtol=config.integrated_rtol,
            max_intervals=config.integrated_max_intervals,
            tolerance=config.integrated_tolerance,
        )
        beneficial = integrated.field[:, 0].detach().cpu().numpy()
        rows.append(
            {
                "jacobian_range": _weighted_distance(jacobian_mass, distance),
                "functional_range": _weighted_distance(functional, distance),
                "jacobian_near": float(jacobian_mass[0]),
                "jacobian_far": float(jacobian_mass[1]),
                "functional_near": float(functional[0]),
                "functional_far": float(functional[1]),
                "beneficial_near": float(beneficial[0]),
                "beneficial_far": float(beneficial[1]),
                "beneficial_sum": float(beneficial.sum()),
                "loss_increase": float(integrated.event_loss_increase[0, 0]),
                "completeness_residual": float(
                    integrated.completeness_residual.abs().max()
                ),
                "quadrature_intervals": float(integrated.intervals.max()),
                "converged": float(integrated.converged.to(DTYPE).mean()),
            }
        )
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
    }


def measure_checkpoints(
    output_dir: Path,
    config: ExperimentConfig,
    *,
    progress: bool = True,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for seed in config.seeds:
        seed_records: list[dict[str, Any]] = []
        for kappa in config.kappas:
            path = _checkpoint_path(output_dir, seed, kappa)
            if not path.exists():
                raise FileNotFoundError(
                    f"missing checkpoint {path}; run --phase train or --phase all first"
                )
            model, checkpoint = _model_from_checkpoint(path, config)
            measured = measure_model(model, config)
            record = {
                "seed": int(seed),
                "kappa": float(kappa),
                "weights": {
                    "near": float(model.near_weight.detach()),
                    "far": float(model.far_weight.detach()),
                },
                "training": checkpoint["training"],
                **measured,
            }
            records.append(record)
            seed_records.append(record)
            if progress:
                print(
                    f"[measure] seed={seed} kappa={kappa:g} "
                    f"rho_J={measured['jacobian_range']:.3f} "
                    f"rho_F={measured['functional_range']:.3f} "
                    f"B={measured['beneficial_near']:+.3f}/"
                    f"{measured['beneficial_far']:+.3f}",
                    flush=True,
                )
        _atomic_json(
            output_dir / "cache" / "measurements" / f"seed_{seed:03d}.json",
            {
                "protocol_version": PROTOCOL_VERSION,
                "fingerprint": config.fingerprint,
                "records": seed_records,
            },
        )

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "fingerprint": config.fingerprint,
        "config": config.record,
        "records": records,
    }
    _atomic_json(output_dir / "results" / "measurements.json", payload)
    _write_measurement_csv(output_dir / "results" / "measurements.csv", records)
    return payload


def _write_measurement_csv(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "seed",
        "kappa",
        "jacobian_range",
        "functional_range",
        "jacobian_near",
        "jacobian_far",
        "functional_near",
        "functional_far",
        "beneficial_near",
        "beneficial_far",
        "beneficial_sum",
        "loss_increase",
        "completeness_residual",
        "quadrature_intervals",
        "converged",
    ]
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record[key] for key in columns})
    os.replace(temporary, path)


def _load_measurements(output_dir: Path, config: ExperimentConfig) -> dict[str, Any]:
    path = output_dir / "results" / "measurements.json"
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path}; run --phase measure or --phase all before figures"
        )
    payload = json.loads(path.read_text())
    if payload.get("fingerprint") != config.fingerprint:
        raise RuntimeError(f"measurement contract mismatch: {path}")
    return payload


def render_figure(output_dir: Path, config: ExperimentConfig) -> dict[str, str]:
    """Render the registered publication figure from cached measurements only."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    payload = _load_measurements(output_dir, config)
    records = payload["records"]
    kappas = np.asarray(sorted({float(row["kappa"]) for row in records}))
    seeds = sorted({int(row["seed"]) for row in records})
    selected_kappa = float(kappas.max())

    def values(field: str, *, kappa: float | None = None) -> np.ndarray:
        selected = records
        if kappa is not None:
            selected = [
                row for row in records if math.isclose(float(row["kappa"]), kappa)
            ]
        return np.asarray([float(row[field]) for row in selected], dtype=np.float64)

    def seed_curve(seed: int, field: str) -> np.ndarray:
        by_kappa = {
            float(row["kappa"]): float(row[field])
            for row in records
            if int(row["seed"]) == seed
        }
        return np.asarray([by_kappa[float(kappa)] for kappa in kappas])

    jacobian = np.stack([seed_curve(seed, "jacobian_range") for seed in seeds])
    functional = np.stack([seed_curve(seed, "functional_range") for seed in seeds])
    exact_range = (
        config.gamma + config.distance * (1.0 + config.gamma)
    ) / (1.0 + 2.0 * config.gamma)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 7.6,
            "axes.linewidth": 0.75,
            "axes.edgecolor": "#333333",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "lines.linewidth": 1.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 400,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )

    blue = "#0072B2"
    orange = "#E69F00"
    green = "#009E73"
    vermillion = "#D55E00"
    ink = "#202020"
    grey = "#7A7A76"
    grid = "#DDDCD8"

    fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.55), constrained_layout=True)
    fig.suptitle(
        "Finite swaps reveal saturated long-range dependence",
        fontsize=10.0,
        fontweight="semibold",
    )

    ax = axes[0]
    for row in jacobian:
        ax.plot(kappas, row, color=blue, alpha=0.18, linewidth=0.8)
    for row in functional:
        ax.plot(kappas, row, color=orange, alpha=0.18, linewidth=0.8)
    ax.plot(
        kappas,
        jacobian.mean(axis=0),
        color=blue,
        marker="o",
        markersize=4.2,
        markerfacecolor="white",
        markeredgewidth=1.2,
        label="Jacobian range",
        zorder=3,
    )
    ax.plot(
        kappas,
        functional.mean(axis=0),
        color=orange,
        marker="s",
        markersize=3.8,
        linestyle=(0, (3.2, 1.8)),
        label="Functional carriage",
        zorder=3,
    )
    ax.axhline(
        exact_range,
        color=grey,
        linestyle=(0, (1.5, 1.8)),
        linewidth=1.2,
        label="Known finite range",
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks(kappas, [f"{value:g}" for value in kappas])
    ax.set_xlabel(r"Saturation strength  $\kappa$")
    ax.set_ylabel("Expected distance (hops)")
    ax.set_title("Range across saturation", pad=5)
    ax.grid(axis="y", color=grid, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower left", handlelength=1.8, labelspacing=0.25)

    ax = axes[1]
    at_kappa = [
        row for row in records if math.isclose(float(row["kappa"]), selected_kappa)
    ]
    jacobian_mass = np.asarray(
        [[row["jacobian_near"], row["jacobian_far"]] for row in at_kappa],
        dtype=np.float64,
    )
    jacobian_share = jacobian_mass / jacobian_mass.sum(axis=1, keepdims=True)
    functional_mass = np.asarray(
        [[row["functional_near"], row["functional_far"]] for row in at_kappa],
        dtype=np.float64,
    )
    functional_share = functional_mass / functional_mass.sum(axis=1, keepdims=True)
    x = np.arange(2)
    width = 0.34
    ax.bar(
        x - width / 2,
        jacobian_share.mean(axis=0),
        width,
        color=blue,
        label="Jacobian",
        zorder=2,
    )
    ax.bar(
        x + width / 2,
        functional_share.mean(axis=0),
        width,
        color=orange,
        label="Finite carriage",
        zorder=2,
    )
    ax.set_xticks(
        x,
        [r"Near  ($d=1$)", rf"Far  ($d={config.distance}$)"],
    )
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("Share of response")
    ax.set_title(rf"Response location at $\kappa={selected_kappa:g}$", pad=5)
    ax.grid(axis="y", color=grid, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper center", labelspacing=0.25)

    ax = axes[2]
    beneficial = np.asarray(
        [[row["beneficial_near"], row["beneficial_far"]] for row in at_kappa],
        dtype=np.float64,
    )
    means = beneficial.mean(axis=0)
    bars = ax.bar(
        x,
        means,
        width=0.55,
        color=(vermillion, green),
        zorder=2,
    )
    ax.axhline(0.0, color=ink, linewidth=0.8)
    total = float(values("beneficial_sum", kappa=selected_kappa).mean())
    loss_increase = float(values("loss_increase", kappa=selected_kappa).mean())
    ax.text(
        0.03,
        0.95,
        rf"$\sum_i B_i = \Delta\mathrm{{MAE}} = {loss_increase:.2f}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.7,
        color=ink,
    )
    ax.set_xticks(
        x,
        [r"Near  ($d=1$)", rf"Far  ($d={config.distance}$)"],
    )
    margin = max(0.5, 0.14 * float(np.max(np.abs(beneficial))))
    ax.set_ylim(float(beneficial.min()) - margin, float(beneficial.max()) + 1.5 * margin)
    ax.set_ylabel("Signed MAE allocation")
    ax.set_title("Task-beneficial carriage", pad=5)
    ax.grid(axis="y", color=grid, linewidth=0.6)
    ax.set_axisbelow(True)
    for bar, label in zip(bars, ("adverse", "beneficial")):
        height = float(bar.get_height())
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height - (0.24 if height >= 0 else -0.24),
            label,
            ha="center",
            va="top" if height >= 0 else "bottom",
            fontsize=7.4,
            color="white",
            fontweight="semibold",
        )
    if not math.isclose(total, loss_increase, abs_tol=config.integrated_tolerance):
        raise RuntimeError("figure data violate Beneficial-carriage completeness")

    for letter, ax in zip(("a", "b", "c"), axes):
        ax.text(
            -0.18,
            1.08,
            letter,
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            va="top",
            ha="left",
            color=ink,
        )

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    png = figure_dir / "finite_intervention_saturation.png"
    pdf = figure_dir / "finite_intervention_saturation.pdf"
    fig.savefig(png)
    fig.savefig(pdf)
    plt.close(fig)

    metadata = {
        "title": "Finite swaps reveal saturated long-range dependence",
        "protocol_version": PROTOCOL_VERSION,
        "fingerprint": config.fingerprint,
        "seeds": seeds,
        "selected_kappa": selected_kappa,
        "known_finite_range": exact_range,
        "beneficial_sum": total,
        "event_loss_increase": loss_increase,
        "files": {"png": str(png), "pdf": str(pdf)},
    }
    _atomic_json(figure_dir / "finite_intervention_saturation.metadata.json", metadata)
    return {"png": str(png), "pdf": str(pdf)}


def run(
    config: ExperimentConfig,
    *,
    output_dir: str | Path,
    phase: str = "all",
    progress: bool = True,
) -> dict[str, Any]:
    output_dir = Path(output_dir).expanduser().resolve()
    ensure_contract(output_dir, config)
    result: dict[str, Any] = {
        "output_dir": str(output_dir),
        "fingerprint": config.fingerprint,
    }
    if phase in {"all", "train"}:
        result["checkpoints"] = [
            str(path) for path in ensure_checkpoints(output_dir, config, progress=progress)
        ]
    if phase in {"all", "measure"}:
        result["measurements"] = measure_checkpoints(
            output_dir,
            config,
            progress=progress,
        )
    if phase in {"all", "figures"}:
        result["figures"] = render_figure(output_dir, config)
    return result


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _parse_csv_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(
        description="Run the four-seed finite-intervention saturation experiment."
    )
    parser.add_argument(
        "--phase",
        choices=("all", "train", "measure", "figures"),
        default="all",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/saturation_carriage_v1",
        help="Drive-backed directory in Colab; local directory otherwise.",
    )
    parser.add_argument("--seeds", default="0,1,2,3")
    parser.add_argument("--kappas", default="0.5,1,2,4,8")
    parser.add_argument("--distance", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--target-alignment", type=float, default=1.0)
    parser.add_argument("--train-points", type=int, default=513)
    parser.add_argument("--train-steps", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--fast-dev-run", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    torch.set_num_threads(max(1, int(args.num_threads)))
    seeds = _parse_csv_ints(args.seeds)
    kappas = _parse_csv_floats(args.kappas)
    train_points = int(args.train_points)
    train_steps = int(args.train_steps)
    if args.fast_dev_run:
        seeds = seeds[:2]
        kappas = tuple(value for value in kappas if value in {0.5, 2.0, 8.0})
        train_points = min(train_points, 129)
        train_steps = min(train_steps, 250)

    config = ExperimentConfig(
        seeds=seeds,
        kappas=kappas,
        distance=args.distance,
        gamma=args.gamma,
        target_alignment=args.target_alignment,
        train_points=train_points,
        train_steps=train_steps,
        learning_rate=args.learning_rate,
    )
    result = run(config, output_dir=args.output_dir, phase=args.phase)
    if "figures" in result:
        print(f"[figure] {result['figures']['png']}", flush=True)
        print(f"[figure] {result['figures']['pdf']}", flush=True)
    return result


if __name__ == "__main__":
    main()
