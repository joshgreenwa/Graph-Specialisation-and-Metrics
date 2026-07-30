"""Learned discrete-routing stress test for finite versus local carriage.

Each graph has a categorical query at source node ``s``, one near carrier at
``d(s, i)=1``, and one record carrier per key at ``d(s, i)=D``. Record order is
random and every record carries a random scalar payload. The target is

    local_code[query] + payload[record whose key matches query].

A tiny model learns a categorical query-key compatibility matrix and routes
record payloads through an ordinary softmax. Valid semantic donor swaps replace
the query with another key already present in the same graph.

The hard-routing teacher gives an eventwise oracle carrier field. We compare its
expected distance with:

* canonical finite Functional carriage; and
* the clean Jacobian projected onto the exact donor-swap direction.

As learned routing becomes nearly one-hot, the softmax Jacobian becomes locally
flat while finite swaps still move between two valid categorical states.
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

from ..methodology.carriage import functional_carriage_events


PROTOCOL_VERSION = "softmax-routing-carriage-v1"
DEFAULT_SEEDS = (0, 1, 2, 3)
DEFAULT_MULTIPLIERS = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0)
DTYPE = torch.float64


@dataclass(frozen=True)
class ExperimentConfig:
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    sharpness_multipliers: tuple[float, ...] = DEFAULT_MULTIPLIERS
    num_keys: int = 6
    far_distance: int = 8
    local_scale: float = 0.75
    remote_scale: float = 1.0
    train_steps: int = 900
    train_batch_size: int = 256
    learning_rate: float = 0.04
    early_stop_mae: float = 3.0e-3
    early_stop_match_probability: float = 0.997
    evaluation_graphs: int = 96
    evaluation_seed: int = 91_337

    def __post_init__(self) -> None:
        if not self.seeds or len(self.seeds) != len(set(self.seeds)):
            raise ValueError("seeds must be non-empty and unique")
        if (
            not self.sharpness_multipliers
            or len(self.sharpness_multipliers)
            != len(set(self.sharpness_multipliers))
        ):
            raise ValueError("sharpness multipliers must be non-empty and unique")
        if any(value <= 0 for value in self.sharpness_multipliers):
            raise ValueError("every sharpness multiplier must be positive")
        if self.num_keys < 3:
            raise ValueError("num_keys must be at least three")
        if self.far_distance <= 1:
            raise ValueError("far_distance must exceed one")
        if self.local_scale <= 0 or self.remote_scale <= 0:
            raise ValueError("local and remote scales must be positive")
        if self.train_steps < 1 or self.train_batch_size < 1:
            raise ValueError("training controls must be positive")
        if self.evaluation_graphs < 1:
            raise ValueError("evaluation_graphs must be positive")

    @property
    def record(self) -> dict[str, Any]:
        return {
            "protocol_version": PROTOCOL_VERSION,
            **asdict(self),
            "seeds": list(self.seeds),
            "sharpness_multipliers": list(self.sharpness_multipliers),
        }

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(self.record, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


class SoftmaxKeyValueRouter(nn.Module):
    """A learned categorical compatibility table with add-pooled carrier states."""

    def __init__(
        self,
        *,
        num_keys: int,
        init_seed: int,
        remote_scale: float = 1.0,
    ) -> None:
        super().__init__()
        generator = torch.Generator(device="cpu").manual_seed(int(init_seed))
        self.num_keys = int(num_keys)
        self.remote_scale = float(remote_scale)
        self.compatibility = nn.Parameter(
            0.05
            * torch.randn(
                self.num_keys,
                self.num_keys,
                generator=generator,
                dtype=DTYPE,
            )
        )
        self.near_weight = nn.Parameter(
            0.05 * torch.randn(self.num_keys, generator=generator, dtype=DTYPE)
        )

    def forward(
        self,
        query_onehot: torch.Tensor,
        record_keys: torch.Tensor,
        payloads: torch.Tensor,
        *,
        sharpness: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return carrier states ``[batch, 1 + records]`` and routing weights."""

        query_onehot = query_onehot.to(dtype=DTYPE)
        payloads = payloads.to(dtype=DTYPE)
        compatibility_by_key = query_onehot @ self.compatibility
        logits = compatibility_by_key.gather(1, record_keys)
        attention = torch.softmax(float(sharpness) * logits, dim=-1)
        near = query_onehot @ self.near_weight
        remote = self.remote_scale * attention * payloads
        states = torch.cat((near.unsqueeze(-1), remote), dim=-1)
        return states, attention


def local_codes(config: ExperimentConfig) -> torch.Tensor:
    return torch.linspace(
        -float(config.local_scale),
        float(config.local_scale),
        config.num_keys,
        dtype=DTYPE,
    )


def sample_graphs(
    *,
    num_graphs: int,
    num_keys: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample permuted key records and independent continuous payloads."""

    order_noise = torch.rand(
        (num_graphs, num_keys),
        generator=generator,
        dtype=DTYPE,
    )
    record_keys = torch.argsort(order_noise, dim=-1)
    payloads = (
        2.0
        * torch.rand(
            (num_graphs, num_keys),
            generator=generator,
            dtype=DTYPE,
        )
        - 1.0
    )
    return record_keys, payloads


def one_hot(indices: torch.Tensor, num_keys: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(indices, num_classes=num_keys).to(dtype=DTYPE)


def teacher_states(
    query_indices: torch.Tensor,
    record_keys: torch.Tensor,
    payloads: torch.Tensor,
    config: ExperimentConfig,
) -> torch.Tensor:
    """Return the known hard-routing near and record carrier states."""

    near = local_codes(config)[query_indices]
    matched = record_keys.eq(query_indices.unsqueeze(-1)).to(dtype=DTYPE)
    remote = float(config.remote_scale) * matched * payloads
    return torch.cat((near.unsqueeze(-1), remote), dim=-1)


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


def _checkpoint_path(output_dir: Path, seed: int) -> Path:
    return output_dir / "cache" / "checkpoints" / f"seed_{int(seed):03d}.pt"


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


def _training_validation_batch(
    config: ExperimentConfig,
    *,
    seed: int,
    num_graphs: int = 512,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(80_003 + int(seed))
    keys, payloads = sample_graphs(
        num_graphs=num_graphs,
        num_keys=config.num_keys,
        generator=generator,
    )
    query_indices = torch.randint(
        config.num_keys,
        (num_graphs,),
        generator=generator,
    )
    query = one_hot(query_indices, config.num_keys)
    target = teacher_states(query_indices, keys, payloads, config).sum(dim=-1)
    return query, keys, payloads, target


def train_one(config: ExperimentConfig, *, seed: int) -> dict[str, Any]:
    """Train one tiny router using graph-level regression supervision only."""

    torch.manual_seed(int(seed))
    generator = torch.Generator(device="cpu").manual_seed(100_003 * int(seed) + 29)
    model = SoftmaxKeyValueRouter(
        num_keys=config.num_keys,
        init_seed=seed,
        remote_scale=config.remote_scale,
    )
    optimiser = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    validation = _training_validation_batch(config, seed=seed)
    history: list[dict[str, float | int]] = []
    started = time.time()

    for step in range(config.train_steps):
        keys, payloads = sample_graphs(
            num_graphs=config.train_batch_size,
            num_keys=config.num_keys,
            generator=generator,
        )
        query_indices = torch.randint(
            config.num_keys,
            (config.train_batch_size,),
            generator=generator,
        )
        query = one_hot(query_indices, config.num_keys)
        target = teacher_states(query_indices, keys, payloads, config).sum(dim=-1)
        states, _ = model(query, keys, payloads)
        loss = torch.nn.functional.mse_loss(states.sum(dim=-1), target)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        should_validate = (
            step == 0
            or (step + 1) % max(1, config.train_steps // 20) == 0
            or step + 1 == config.train_steps
        )
        if should_validate:
            with torch.no_grad():
                val_query, val_keys, val_payloads, val_target = validation
                val_states, val_attention = model(val_query, val_keys, val_payloads)
                val_prediction = val_states.sum(dim=-1)
                matched = val_keys.eq(val_query.argmax(dim=-1, keepdim=True))
                match_probability = float(
                    val_attention.masked_select(matched).mean()
                )
                val_mae = float((val_prediction - val_target).abs().mean())
            history.append(
                {
                    "step": int(step + 1),
                    "train_mse": float(loss.detach()),
                    "validation_mae": val_mae,
                    "match_probability": match_probability,
                }
            )
            if (
                step >= 100
                and val_mae <= config.early_stop_mae
                and match_probability >= config.early_stop_match_probability
            ):
                break

    final = history[-1]
    return {
        "protocol_version": PROTOCOL_VERSION,
        "fingerprint": config.fingerprint,
        "seed": int(seed),
        "state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "training": {
            "steps": int(step + 1),
            "validation_mae": float(final["validation_mae"]),
            "match_probability": float(final["match_probability"]),
            "seconds": round(time.time() - started, 4),
            "history": history,
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
        path = _checkpoint_path(output_dir, seed)
        if path.exists():
            payload = _load_torch(path)
            if payload.get("fingerprint") != config.fingerprint:
                raise RuntimeError(f"checkpoint contract mismatch: {path}")
            action = "reuse"
        else:
            payload = train_one(config, seed=seed)
            _atomic_torch(path, payload)
            action = "train"
        saved.append(path)
        if progress:
            training = payload["training"]
            print(
                f"[{action}] seed={seed} steps={training['steps']} "
                f"MAE={training['validation_mae']:.3e} "
                f"p(match)={training['match_probability']:.4f}",
                flush=True,
            )
    return saved


def _model_from_checkpoint(
    path: Path,
    config: ExperimentConfig,
) -> tuple[SoftmaxKeyValueRouter, dict[str, Any]]:
    payload = _load_torch(path)
    if payload.get("fingerprint") != config.fingerprint:
        raise RuntimeError(f"checkpoint contract mismatch: {path}")
    model = SoftmaxKeyValueRouter(
        num_keys=config.num_keys,
        init_seed=0,
        remote_scale=config.remote_scale,
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def _evaluation_sources(
    config: ExperimentConfig,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    generator = torch.Generator(device="cpu").manual_seed(config.evaluation_seed)
    base_keys, base_payloads = sample_graphs(
        num_graphs=config.evaluation_graphs,
        num_keys=config.num_keys,
        generator=generator,
    )
    source_indices = torch.arange(config.num_keys).repeat(config.evaluation_graphs)
    record_keys = base_keys.repeat_interleave(config.num_keys, dim=0)
    payloads = base_payloads.repeat_interleave(config.num_keys, dim=0)
    source_query = one_hot(source_indices, config.num_keys)

    donor_rows = [
        [key for key in range(config.num_keys) if key != source]
        for source in source_indices.tolist()
    ]
    donor_indices = torch.tensor(donor_rows, dtype=torch.long)
    donor_query = one_hot(donor_indices, config.num_keys)
    return source_indices, source_query, donor_indices, donor_query, record_keys, payloads


def _event_ranges(mass: torch.Tensor, distances: torch.Tensor) -> torch.Tensor:
    total = mass.sum(dim=-1)
    weighted = (mass * distances).sum(dim=-1)
    return torch.where(
        total > torch.finfo(mass.dtype).tiny,
        weighted / total,
        torch.full_like(total, float("nan")),
    )


def _near_far_share(mass: torch.Tensor) -> tuple[float, float]:
    near = mass[..., 0].sum()
    far = mass[..., 1:].sum()
    total = near + far
    if not bool(torch.isfinite(total)) or float(total) <= 0:
        return float("nan"), float("nan")
    return float(near / total), float(far / total)


def measure_model(
    model: SoftmaxKeyValueRouter,
    config: ExperimentConfig,
) -> list[dict[str, float]]:
    """Measure all confidence settings on one fixed set of valid donor swaps."""

    (
        source_indices,
        source_query,
        donor_indices,
        donor_query,
        record_keys,
        payloads,
    ) = _evaluation_sources(config)
    sources, donors, _ = donor_query.shape
    carriers = 1 + config.num_keys
    distances = torch.tensor(
        [1.0] + [float(config.far_distance)] * config.num_keys,
        dtype=DTYPE,
    )

    teacher_clean = teacher_states(
        source_indices,
        record_keys,
        payloads,
        config,
    )
    repeated_keys = record_keys.unsqueeze(1).expand(-1, donors, -1)
    repeated_payloads = payloads.unsqueeze(1).expand(-1, donors, -1)
    teacher_event = teacher_states(
        donor_indices.reshape(-1),
        repeated_keys.reshape(-1, config.num_keys),
        repeated_payloads.reshape(-1, config.num_keys),
        config,
    ).reshape(sources, donors, carriers)
    oracle_mass = (teacher_clean.unsqueeze(1) - teacher_event).abs()
    oracle_range_events = _event_ranges(oracle_mass, distances)
    oracle_near_share, oracle_far_share = _near_far_share(oracle_mass)

    records: list[dict[str, float]] = []
    for multiplier in config.sharpness_multipliers:
        clean_query = source_query.detach().clone().requires_grad_(True)
        clean_states, attention = model(
            clean_query,
            record_keys,
            payloads,
            sharpness=multiplier,
        )

        gradients: list[torch.Tensor] = []
        for carrier in range(carriers):
            gradient = torch.autograd.grad(
                clean_states[:, carrier].sum(),
                clean_query,
                retain_graph=carrier + 1 < carriers,
            )[0]
            gradients.append(gradient)
        state_jacobian = torch.stack(gradients, dim=1)
        direction = clean_query.detach().unsqueeze(1) - donor_query
        jacobian_mass = torch.einsum(
            "snk,sdk->sdn",
            state_jacobian,
            direction,
        ).abs()

        with torch.no_grad():
            event_states, _ = model(
                donor_query.reshape(-1, config.num_keys),
                repeated_keys.reshape(-1, config.num_keys),
                repeated_payloads.reshape(-1, config.num_keys),
                sharpness=multiplier,
            )
            event_states = event_states.reshape(sources, donors, carriers)
            delta = clean_states.detach().unsqueeze(1) - event_states
            output_gradient = torch.ones((1, carriers, 1), dtype=DTYPE)
            functional_mass = functional_carriage_events(
                delta.unsqueeze(-1),
                output_gradient,
            )

            functional_range_events = _event_ranges(functional_mass, distances)
            jacobian_range_events = _event_ranges(jacobian_mass, distances)
            prediction = clean_states.detach().sum(dim=-1)
            target = teacher_clean.sum(dim=-1)
            matched = record_keys.eq(source_indices.unsqueeze(-1))
            matched_probability = float(attention.masked_select(matched).mean())
            entropy = -(
                attention.clamp_min(torch.finfo(DTYPE).tiny)
                * attention.clamp_min(torch.finfo(DTYPE).tiny).log()
            ).sum(dim=-1)
            normalized_entropy = float((entropy / math.log(config.num_keys)).mean())

            finite_near_share, finite_far_share = _near_far_share(functional_mass)
            jacobian_near_share, jacobian_far_share = _near_far_share(jacobian_mass)
            output_difference = (
                prediction.unsqueeze(1) - event_states.sum(dim=-1)
            )
            completeness_error = float(
                (delta.sum(dim=-1) - output_difference).abs().max()
            )

        records.append(
            {
                "sharpness_multiplier": float(multiplier),
                "matched_probability": matched_probability,
                "normalized_attention_entropy": normalized_entropy,
                "test_mae": float((prediction - target).abs().mean()),
                "jacobian_range": float(torch.nanmean(jacobian_range_events)),
                "functional_range": float(torch.nanmean(functional_range_events)),
                "oracle_range": float(torch.nanmean(oracle_range_events)),
                "jacobian_oracle_range_mae": float(
                    torch.nanmean((jacobian_range_events - oracle_range_events).abs())
                ),
                "functional_oracle_range_mae": float(
                    torch.nanmean((functional_range_events - oracle_range_events).abs())
                ),
                "jacobian_near_share": jacobian_near_share,
                "jacobian_far_share": jacobian_far_share,
                "functional_near_share": finite_near_share,
                "functional_far_share": finite_far_share,
                "oracle_near_share": oracle_near_share,
                "oracle_far_share": oracle_far_share,
                "output_completeness_error": completeness_error,
            }
        )
    return records


def measure_checkpoints(
    output_dir: Path,
    config: ExperimentConfig,
    *,
    progress: bool = True,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for seed in config.seeds:
        path = _checkpoint_path(output_dir, seed)
        if not path.exists():
            raise FileNotFoundError(
                f"missing checkpoint {path}; run --phase train or --phase all first"
            )
        model, checkpoint = _model_from_checkpoint(path, config)
        seed_records = [
            {
                "seed": int(seed),
                "training": checkpoint["training"],
                **record,
            }
            for record in measure_model(model, config)
        ]
        records.extend(seed_records)
        _atomic_json(
            output_dir / "cache" / "measurements" / f"seed_{seed:03d}.json",
            {
                "protocol_version": PROTOCOL_VERSION,
                "fingerprint": config.fingerprint,
                "records": seed_records,
            },
        )
        if progress:
            final = seed_records[-1]
            print(
                f"[measure] seed={seed} p(match)={final['matched_probability']:.4f} "
                f"rho_J={final['jacobian_range']:.3f} "
                f"rho_F={final['functional_range']:.3f} "
                f"rho_oracle={final['oracle_range']:.3f}",
                flush=True,
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
        "sharpness_multiplier",
        "matched_probability",
        "normalized_attention_entropy",
        "test_mae",
        "jacobian_range",
        "functional_range",
        "oracle_range",
        "jacobian_oracle_range_mae",
        "functional_oracle_range_mae",
        "jacobian_near_share",
        "jacobian_far_share",
        "functional_near_share",
        "functional_far_share",
        "oracle_near_share",
        "oracle_far_share",
        "output_completeness_error",
    ]
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow({column: record[column] for column in columns})
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
    """Render the registered figure entirely from cached measurements."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    payload = _load_measurements(output_dir, config)
    records = payload["records"]
    multipliers = np.asarray(
        sorted({float(record["sharpness_multiplier"]) for record in records})
    )
    seeds = sorted({int(record["seed"]) for record in records})

    def seed_curve(seed: int, field: str) -> np.ndarray:
        values = {
            float(record["sharpness_multiplier"]): float(record[field])
            for record in records
            if int(record["seed"]) == seed
        }
        return np.asarray([values[float(value)] for value in multipliers])

    confidence = np.stack(
        [seed_curve(seed, "matched_probability") for seed in seeds]
    )
    test_mae = np.stack([seed_curve(seed, "test_mae") for seed in seeds])
    jacobian_range = np.stack(
        [seed_curve(seed, "jacobian_range") for seed in seeds]
    )
    functional_range = np.stack(
        [seed_curve(seed, "functional_range") for seed in seeds]
    )
    oracle_range = np.stack([seed_curve(seed, "oracle_range") for seed in seeds])

    high_records = [
        record
        for record in records
        if math.isclose(
            float(record["sharpness_multiplier"]),
            float(multipliers.max()),
        )
    ]
    high_jacobian = np.asarray(
        [float(record["jacobian_range"]) for record in high_records],
        dtype=np.float64,
    )
    high_functional = np.asarray(
        [float(record["functional_range"]) for record in high_records],
        dtype=np.float64,
    )
    high_oracle = np.asarray(
        [float(record["oracle_range"]) for record in high_records],
        dtype=np.float64,
    )
    high_test_mae = float(np.mean([record["test_mae"] for record in high_records]))

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 7.5,
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
    purple = "#CC79A7"
    grey = "#7A7A76"
    light_blue = "#56B4E9"
    yellow = "#F0E442"
    ink = "#202020"
    grid = "#DDDCD8"
    pale = "#F4F3EF"

    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.65), constrained_layout=True)
    fig.suptitle(
        "Range estimation in a learned softmax-routing task",
        fontsize=10.0,
        fontweight="semibold",
    )

    ax = axes[0]
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    ax.text(
        0.12,
        0.62,
        "Query source\nkey A",
        ha="center",
        va="center",
        fontsize=7.7,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": yellow,
            "edgecolor": ink,
            "linewidth": 0.8,
        },
    )
    ax.text(
        0.48,
        0.78,
        "Near carrier\n$c_A$",
        ha="center",
        va="center",
        fontsize=7.6,
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": light_blue,
            "edgecolor": ink,
            "linewidth": 0.8,
        },
    )
    ax.text(
        0.83,
        0.73,
        "Matched far record\nkey A; payload $v_A$",
        ha="center",
        va="center",
        fontsize=7.0,
        color="white",
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": purple,
            "edgecolor": ink,
            "linewidth": 0.8,
        },
    )
    ax.text(
        0.84,
        0.39,
        "Other records\nkeys B–F",
        ha="center",
        va="center",
        fontsize=7.2,
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "#E7E6E2",
            "edgecolor": grey,
            "linewidth": 0.8,
        },
    )
    for end in ((0.39, 0.73), (0.73, 0.70), (0.73, 0.43)):
        ax.annotate(
            "",
            xy=end,
            xytext=(0.21, 0.62),
            arrowprops={
                "arrowstyle": "-|>",
                "color": grey,
                "linewidth": 1.0,
                "shrinkA": 2,
                "shrinkB": 2,
            },
        )
    ax.text(0.29, 0.72, "$d=1$", fontsize=7.2, color=ink, ha="center")
    ax.text(0.56, 0.61, f"$d={config.far_distance}$", fontsize=7.2, color=ink)
    ax.text(
        0.5,
        0.18,
        r"Known target:  $y=c_A+v_A$",
        ha="center",
        va="center",
        fontsize=8.0,
        fontweight="semibold",
    )
    ax.text(
        0.5,
        0.07,
        r"Valid swap $A\!\rightarrow\!B$ selects record B",
        ha="center",
        va="center",
        fontsize=7.2,
        color=grey,
    )
    ax.set_title("Task and ground truth", pad=5)

    ax = axes[1]
    ax.axvspan(0.95, 1.005, color=pale, zorder=0)
    for values, color in (
        (jacobian_range, blue),
        (functional_range, orange),
    ):
        for seed_index in range(len(seeds)):
            ax.plot(
                confidence[seed_index],
                values[seed_index],
                color=color,
                alpha=0.18,
                linewidth=0.8,
            )
    ax.plot(
        confidence.mean(axis=0),
        jacobian_range.mean(axis=0),
        color=blue,
        marker="o",
        markersize=4.1,
        markerfacecolor="white",
        markeredgewidth=1.2,
        label="Local Jacobian estimate",
    )
    ax.plot(
        confidence.mean(axis=0),
        functional_range.mean(axis=0),
        color=orange,
        marker="s",
        markersize=3.8,
        linestyle=(0, (3.2, 1.8)),
        label="Finite-carriage estimate",
    )
    ax.plot(
        confidence.mean(axis=0),
        oracle_range.mean(axis=0),
        color=grey,
        linestyle=(0, (1.5, 1.8)),
        linewidth=1.2,
        label="Known target range",
    )
    ax.set_xlim(max(0.0, float(confidence.min()) - 0.035), 1.005)
    ax.set_xlabel("Routing confidence  $p$(matched record)")
    ax.set_ylabel("Estimated range (hops)")
    ax.set_title("Range across routing confidence", pad=5)
    ax.grid(axis="y", color=grid, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="best", handlelength=1.8, labelspacing=0.25)
    ax.text(
        0.97,
        0.53,
        f"At rightmost point:\nmean test MAE = {high_test_mae:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=7.0,
        color=ink,
        bbox={
            "boxstyle": "round,pad=0.2",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.86,
        },
    )

    ax = axes[2]
    x = np.arange(2)
    method_values = np.asarray(
        [high_jacobian.mean(), high_functional.mean()],
        dtype=np.float64,
    )
    bars = ax.bar(
        x,
        method_values,
        width=0.58,
        color=(blue, orange),
        zorder=2,
    )
    target_range = float(high_oracle.mean())
    ax.axhline(
        target_range,
        color=grey,
        linestyle=(0, (1.5, 1.8)),
        linewidth=1.4,
        zorder=3,
    )
    offsets = np.linspace(-0.055, 0.055, len(seeds))
    for method, seed_values in enumerate((high_jacobian, high_functional)):
        ax.scatter(
            method + offsets,
            seed_values,
            s=13,
            facecolor="white",
            edgecolor=ink,
            linewidth=0.6,
            zorder=4,
        )
    for method, (bar, value) in enumerate(zip(bars, method_values)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.12 if method == 0 else value - 0.22,
            f"{value:.2f}",
            ha="center",
            va="bottom" if method == 0 else "top",
            fontsize=7.8,
            color=ink if method == 0 else "white",
            fontweight="semibold",
        )
    ax.text(
        0.03,
        target_range + 0.08,
        f"Known target: {target_range:.2f} hops",
        transform=ax.get_yaxis_transform(),
        ha="left",
        va="bottom",
        fontsize=7.2,
        color=grey,
    )
    ax.set_xticks(x, ["Local Jacobian\nestimate", "Finite-carriage\nestimate"])
    ax.set_ylim(0.0, max(5.8, target_range + 0.55))
    ax.set_ylabel("Estimated range (hops)")
    ax.set_title("High-confidence comparison", pad=5)
    ax.grid(axis="y", color=grid, linewidth=0.6)
    ax.set_axisbelow(True)

    for letter, ax in zip(("a", "b", "c"), axes):
        ax.text(
            -0.13,
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
    png = figure_dir / "softmax_routing_carriage.png"
    pdf = figure_dir / "softmax_routing_carriage.pdf"
    fig.savefig(png)
    fig.savefig(pdf)
    plt.close(fig)

    metadata = {
        "title": "Range estimation in a learned softmax-routing task",
        "protocol_version": PROTOCOL_VERSION,
        "fingerprint": config.fingerprint,
        "seeds": seeds,
        "highest_sharpness_multiplier": float(multipliers.max()),
        "highest_matched_probability": float(confidence[:, -1].mean()),
        "files": {"png": str(png), "pdf": str(pdf)},
    }
    _atomic_json(figure_dir / "softmax_routing_carriage.metadata.json", metadata)
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
            str(path)
            for path in ensure_checkpoints(output_dir, config, progress=progress)
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
        description="Run the learned softmax-routing carriage experiment."
    )
    parser.add_argument(
        "--phase",
        choices=("all", "train", "measure", "figures"),
        default="all",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/softmax_routing_carriage_v1",
        help="Drive-backed directory in Colab; local directory otherwise.",
    )
    parser.add_argument("--seeds", default="0,1,2,3")
    parser.add_argument("--sharpness-multipliers", default="0.125,0.25,0.5,1,2,4")
    parser.add_argument("--num-keys", type=int, default=6)
    parser.add_argument("--far-distance", type=int, default=8)
    parser.add_argument("--local-scale", type=float, default=0.75)
    parser.add_argument("--remote-scale", type=float, default=1.0)
    parser.add_argument("--train-steps", type=int, default=900)
    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--evaluation-graphs", type=int, default=96)
    parser.add_argument("--evaluation-seed", type=int, default=91_337)
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--fast-dev-run", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    torch.set_num_threads(max(1, int(args.num_threads)))
    seeds = _parse_csv_ints(args.seeds)
    multipliers = _parse_csv_floats(args.sharpness_multipliers)
    train_steps = int(args.train_steps)
    train_batch_size = int(args.train_batch_size)
    evaluation_graphs = int(args.evaluation_graphs)
    if args.fast_dev_run:
        seeds = seeds[:2]
        multipliers = tuple(
            value for value in multipliers if value in {0.25, 1.0, 4.0}
        )
        train_steps = min(train_steps, 150)
        train_batch_size = min(train_batch_size, 96)
        evaluation_graphs = min(evaluation_graphs, 16)

    config = ExperimentConfig(
        seeds=seeds,
        sharpness_multipliers=multipliers,
        num_keys=args.num_keys,
        far_distance=args.far_distance,
        local_scale=args.local_scale,
        remote_scale=args.remote_scale,
        train_steps=train_steps,
        train_batch_size=train_batch_size,
        learning_rate=args.learning_rate,
        evaluation_graphs=evaluation_graphs,
        evaluation_seed=args.evaluation_seed,
    )
    result = run(config, output_dir=args.output_dir, phase=args.phase)
    if "figures" in result:
        print(f"[figure] {result['figures']['png']}", flush=True)
        print(f"[figure] {result['figures']['pdf']}", flush=True)
    return result


if __name__ == "__main__":
    main()
