"""Matched additive-versus-gated experiment with a learned softmax router.

The four carriers are records in a two-bank, two-key memory.  Structural
context selects a bank and semantic context selects a key.  The gated teacher
routes to their conjunction.  Its additive control shares the clean,
semantic-donor, and structural-donor fields exactly, but defines the joint
donor as the additive completion of those three fields.

Consequently, ordinary semantic and structural distance profiles are exactly
matched between the tasks.  Only the finite semantic-by-structural interaction
distinguishes conjunctive routing from the additive null.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

DTYPE = torch.float64

# Rows: clean (q0, r0), semantic donor (q1, r0), structural donor
# (q0, r1), and joint donor (q1, r1).  Columns concatenate one-hot query and
# structural context encodings.
EVENT_INPUTS = torch.tensor(
    (
        (1.0, 0.0, 1.0, 0.0),
        (0.0, 1.0, 1.0, 0.0),
        (1.0, 0.0, 0.0, 1.0),
        (0.0, 1.0, 0.0, 1.0),
    ),
    dtype=DTYPE,
)

# Carrier order: (bank 0, key 0), (bank 0, key 1),
# (bank 1, key 0), (bank 1, key 1).
CARRIER_DISTANCES = np.asarray((1, 4, 4, 6), dtype=np.int64)
UNIQUE_DISTANCES = np.asarray((1, 4, 6), dtype=np.int64)
RECORD_VALUES = torch.tensor((1.0, 1.5, 1.5, 4.0), dtype=DTYPE)


@dataclass(frozen=True)
class ExperimentConfig:
    seeds: tuple[int, ...] = tuple(range(8))
    hidden_dim: int = 24
    heads: int = 4
    learning_rate: float = 5.0e-3
    max_steps: int = 8_000
    early_stop_mse: float = 1.0e-12
    interaction_effect_floor: float = 1.0e-4
    interaction_scale: float = 1.0

    def validate(self) -> None:
        if not self.seeds or len(self.seeds) != len(set(self.seeds)):
            raise ValueError("seeds must be non-empty and unique")
        if self.hidden_dim < 2 or self.heads < 1 or self.max_steps < 1:
            raise ValueError("hidden_dim, heads, and max_steps must be positive")
        if self.learning_rate <= 0 or self.early_stop_mse <= 0:
            raise ValueError("learning rate and early-stop threshold must be positive")
        if self.interaction_effect_floor <= 0:
            raise ValueError("interaction_effect_floor must be positive")
        if not 0 <= self.interaction_scale <= 1:
            raise ValueError("interaction_scale must lie in [0, 1]")


def teacher_contributions(*, task: str, interaction_scale: float = 1.0) -> torch.Tensor:
    """Return node/carrier-supervised fields for the four intervention states."""

    gated = torch.diag(RECORD_VALUES)
    if task == "additive":
        scale = 0.0
    elif task == "gated":
        scale = float(interaction_scale)
        if not 0 <= scale <= 1:
            raise ValueError("interaction_scale must lie in [0, 1]")
    else:
        raise ValueError(f"unknown task {task!r}")

    additive = gated.clone()
    additive[3] = gated[1] + gated[2] - gated[0]
    return additive + scale * (gated - additive)


class MultiHeadSoftmaxRouter(nn.Module):
    """Tiny attention-like router with signed per-head message amplitudes."""

    def __init__(self, config: ExperimentConfig, *, seed: int) -> None:
        super().__init__()
        torch.manual_seed(int(seed))
        self.heads = int(config.heads)
        self.carriers = int(CARRIER_DISTANCES.size)
        self.encoder = nn.Sequential(
            nn.Linear(4, int(config.hidden_dim), dtype=DTYPE),
            nn.Tanh(),
            nn.Linear(int(config.hidden_dim), int(config.hidden_dim), dtype=DTYPE),
            nn.Tanh(),
        )
        self.logit_head = nn.Linear(
            int(config.hidden_dim), self.heads * self.carriers, dtype=DTYPE
        )
        self.amplitude_head = nn.Linear(
            int(config.hidden_dim), self.heads, dtype=DTYPE
        )

    def route(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(inputs.to(dtype=DTYPE))
        logits = self.logit_head(hidden).reshape(-1, self.heads, self.carriers)
        probabilities = torch.softmax(logits, dim=-1)
        amplitudes = self.amplitude_head(hidden)
        return probabilities, amplitudes

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        probabilities, amplitudes = self.route(inputs)
        return torch.sum(probabilities * amplitudes.unsqueeze(-1), dim=1)


def train_model(
    config: ExperimentConfig,
    *,
    task: str,
    seed: int,
) -> tuple[MultiHeadSoftmaxRouter, dict[str, float | int]]:
    target = teacher_contributions(
        task=task, interaction_scale=config.interaction_scale
    )
    model = MultiHeadSoftmaxRouter(config, seed=int(seed))
    optimiser = torch.optim.Adam(model.parameters(), lr=float(config.learning_rate))
    final_mse = float("inf")
    step = 0
    for step in range(1, int(config.max_steps) + 1):
        prediction = model(EVENT_INPUTS)
        loss = torch.mean((prediction - target) ** 2)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        final_mse = float(loss.detach())
        if final_mse <= float(config.early_stop_mse):
            break
    model.eval()
    with torch.no_grad():
        prediction = model(EVENT_INPUTS)
        error = torch.abs(prediction - target)
        probabilities, _ = model.route(EVENT_INPUTS)
        routing_entropy = -torch.sum(
            probabilities * torch.log(probabilities.clamp_min(1.0e-12)), dim=-1
        ).mean()
    return model, {
        "steps": int(step),
        "mse": float(torch.mean((prediction - target) ** 2)),
        "max_abs_error": float(error.max()),
        "mean_head_entropy": float(routing_entropy),
    }


def _distance_profile(response: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    mass = torch.abs(response).detach().cpu().numpy()
    grouped = np.asarray(
        [mass[CARRIER_DISTANCES == distance].sum() for distance in UNIQUE_DISTANCES],
        dtype=np.float64,
    )
    total = float(grouped.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("cannot normalise a null or non-finite response")
    return grouped, grouped / total


def measure_four_state_field(
    contributions: torch.Tensor,
    *,
    effect_floor: float = 1.0e-5,
) -> dict[str, Any]:
    """Measure ordinary marginals and the finite joint-intervention contrast."""

    if tuple(contributions.shape) != (4, 4):
        raise ValueError("contributions must have shape [four states, four carriers]")
    clean, semantic, structural, joint = contributions
    semantic_response = clean - semantic
    structural_response = clean - structural
    interaction = clean - semantic - structural + joint
    semantic_mass, semantic_profile = _distance_profile(semantic_response)
    structural_mass, structural_profile = _distance_profile(structural_response)

    carrier_interaction_mass = torch.abs(interaction).detach().cpu().numpy()
    distance_interaction_mass = np.asarray(
        [
            carrier_interaction_mass[CARRIER_DISTANCES == distance].sum()
            for distance in UNIQUE_DISTANCES
        ],
        dtype=np.float64,
    )
    interaction_mass = float(distance_interaction_mass.sum())
    interaction_estimable = bool(
        np.isfinite(interaction_mass) and interaction_mass > float(effect_floor)
    )
    interaction_profile = (
        distance_interaction_mass / interaction_mass
        if interaction_estimable
        else np.full_like(distance_interaction_mass, np.nan)
    )
    far_mask = UNIQUE_DISTANCES >= 4
    return {
        "semantic_response": semantic_response.detach().cpu().numpy(),
        "structural_response": structural_response.detach().cpu().numpy(),
        "semantic_distance_mass": semantic_mass,
        "structural_distance_mass": structural_mass,
        "semantic_distance_profile": semantic_profile,
        "structural_distance_profile": structural_profile,
        "marginal_profile_tv": float(
            0.5 * np.abs(semantic_profile - structural_profile).sum()
        ),
        "carrier_interaction": interaction.detach().cpu().numpy(),
        "output_interaction": float(interaction.sum()),
        "interaction_mass": interaction_mass,
        "interaction_estimable": interaction_estimable,
        "interaction_distance_profile": interaction_profile,
        "interaction_expected_distance": (
            float(np.dot(interaction_profile, UNIQUE_DISTANCES))
            if interaction_estimable
            else float("nan")
        ),
        "far_interaction_share": (
            float(interaction_profile[far_mask].sum())
            if interaction_estimable
            else float("nan")
        ),
    }


def _serialise(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _serialise(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialise(item) for item in value]
    return value


def run(config: ExperimentConfig | None = None) -> dict[str, Any]:
    """Train the matched tasks over seeds and summarise recovered contrasts."""

    config = ExperimentConfig() if config is None else config
    config.validate()
    torch.set_num_threads(1)
    result: dict[str, Any] = {
        "config": asdict(config),
        "carrier_distances": CARRIER_DISTANCES,
        "unique_distances": UNIQUE_DISTANCES,
        "tasks": {},
    }
    teacher_measurements: dict[str, dict[str, Any]] = {}
    for task in ("additive", "gated"):
        teacher = teacher_contributions(
            task=task, interaction_scale=config.interaction_scale
        )
        teacher_measurement = measure_four_state_field(
            teacher, effect_floor=config.interaction_effect_floor
        )
        teacher_measurements[task] = teacher_measurement
        seeds: list[dict[str, Any]] = []
        for seed in config.seeds:
            model, training = train_model(config, task=task, seed=int(seed))
            with torch.no_grad():
                learned = model(EVENT_INPUTS)
            measurement = measure_four_state_field(
                learned, effect_floor=config.interaction_effect_floor
            )
            seeds.append(
                {
                    "seed": int(seed),
                    "training": training,
                    "measurement": measurement,
                }
            )
        estimable = [
            item["measurement"]
            for item in seeds
            if item["measurement"]["interaction_estimable"]
        ]
        result["tasks"][task] = {
            "teacher": teacher_measurement,
            "seeds": seeds,
            "summary": {
                "max_fit_error": float(
                    max(item["training"]["max_abs_error"] for item in seeds)
                ),
                "mean_output_interaction": float(
                    np.mean(
                        [item["measurement"]["output_interaction"] for item in seeds]
                    )
                ),
                "max_abs_output_interaction": float(
                    max(
                        abs(item["measurement"]["output_interaction"])
                        for item in seeds
                    )
                ),
                "min_interaction_mass": float(
                    min(item["measurement"]["interaction_mass"] for item in seeds)
                ),
                "max_interaction_mass": float(
                    max(item["measurement"]["interaction_mass"] for item in seeds)
                ),
                "max_carrier_interaction_error": float(
                    max(
                        np.max(
                            np.abs(
                                np.asarray(
                                    item["measurement"]["carrier_interaction"]
                                )
                                - np.asarray(
                                    teacher_measurement["carrier_interaction"]
                                )
                            )
                        )
                        for item in seeds
                    )
                ),
                "estimable_interaction_seeds": len(estimable),
                "mean_far_interaction_share": (
                    float(np.mean([item["far_interaction_share"] for item in estimable]))
                    if estimable
                    else None
                ),
                "mean_interaction_expected_distance": (
                    float(
                        np.mean(
                            [item["interaction_expected_distance"] for item in estimable]
                        )
                    )
                    if estimable
                    else None
                ),
                "max_marginal_profile_tv": float(
                    max(
                        item["measurement"]["marginal_profile_tv"] for item in seeds
                    )
                ),
                "mean_head_entropy": float(
                    np.mean([item["training"]["mean_head_entropy"] for item in seeds])
                ),
            },
        }
    result["teacher_semantic_profile_tv_between_tasks"] = float(
        0.5
        * np.abs(
            teacher_measurements["additive"]["semantic_distance_profile"]
            - teacher_measurements["gated"]["semantic_distance_profile"]
        ).sum()
    )
    result["teacher_structural_profile_tv_between_tasks"] = float(
        0.5
        * np.abs(
            teacher_measurements["additive"]["structural_distance_profile"]
            - teacher_measurements["gated"]["structural_distance_profile"]
        ).sum()
    )
    additive_seeds = result["tasks"]["additive"]["seeds"]
    gated_seeds = result["tasks"]["gated"]["seeds"]
    result["learned_profile_match"] = {
        "max_semantic_tv_between_tasks": float(
            max(
                0.5
                * np.abs(
                    np.asarray(
                        additive["measurement"]["semantic_distance_profile"]
                    )
                    - np.asarray(gated["measurement"]["semantic_distance_profile"])
                ).sum()
                for additive, gated in zip(additive_seeds, gated_seeds, strict=True)
            )
        ),
        "max_structural_tv_between_tasks": float(
            max(
                0.5
                * np.abs(
                    np.asarray(
                        additive["measurement"]["structural_distance_profile"]
                    )
                    - np.asarray(gated["measurement"]["structural_distance_profile"])
                ).sum()
                for additive, gated in zip(additive_seeds, gated_seeds, strict=True)
            )
        ),
    }
    return _serialise(result)


def _parse_seeds(raw: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if not seeds:
        raise argparse.ArgumentTypeError("provide at least one integer seed")
    return seeds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=_parse_seeds, default=tuple(range(8)))
    parser.add_argument("--max-steps", type=int, default=8_000)
    parser.add_argument("--hidden-dim", type=int, default=24)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--interaction-scale", type=float, default=1.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ExperimentConfig(
        seeds=tuple(args.seeds),
        max_steps=int(args.max_steps),
        hidden_dim=int(args.hidden_dim),
        heads=int(args.heads),
        interaction_scale=float(args.interaction_scale),
    )
    print(json.dumps(run(config), indent=2, sort_keys=True, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
