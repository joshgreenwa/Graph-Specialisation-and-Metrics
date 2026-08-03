"""Tiny additive-versus-structure-gated carrier-response experiment.

The experiment is deliberately minimal.  A semantic bit ``s`` and structural
context bit ``t`` determine contributions at two registered graph-distance
carriers.  The additive and gated teachers are constructed to have exactly the
same semantic and structural marginal response at the canonical donor event,
while only the gated teacher has a semantic--structural interaction.

This isolates the proposed conditional-carriage estimand from graph generation,
checkpoint loading, and long training runs.  A two-layer MLP is trained on the
four categorical states and evaluated with the same finite 2x2 contrast that
would be used for a graph model.
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
STATES = torch.tensor(
    ((1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)),
    dtype=DTYPE,
)
DISTANCES = np.asarray((1, 4), dtype=np.int64)


@dataclass(frozen=True)
class ExperimentConfig:
    seeds: tuple[int, ...] = tuple(range(8))
    hidden_dim: int = 12
    learning_rate: float = 3.0e-2
    max_steps: int = 4_000
    early_stop_mse: float = 1.0e-12
    interaction_strength: float = 0.75
    interaction_effect_floor: float = 1.0e-5

    def validate(self) -> None:
        if not self.seeds or len(self.seeds) != len(set(self.seeds)):
            raise ValueError("seeds must be non-empty and unique")
        if self.hidden_dim < 2 or self.max_steps < 1:
            raise ValueError("hidden_dim and max_steps must be positive")
        if (
            self.learning_rate <= 0
            or self.early_stop_mse <= 0
            or self.interaction_effect_floor <= 0
        ):
            raise ValueError("learning rate and early-stop threshold must be positive")
        if not 0 < self.interaction_strength < 1.5:
            raise ValueError("interaction_strength must lie in (0, 1.5)")


def teacher_contributions(
    states: torch.Tensor,
    *,
    task: str,
    interaction_strength: float,
) -> torch.Tensor:
    """Return ``[..., carrier=(near, far)]`` teacher contributions.

    Both tasks have the canonical clean-to-donor marginal field ``(2, 3)`` for
    semantic and structural flips.  The gated task adds ``gamma * s * t`` at
    the far carrier while reducing its additive coefficient by ``gamma``; this
    preserves those clean-event marginals exactly.  Its finite 2x2 interaction
    is therefore ``(0, 4*gamma)``.
    """

    if states.shape[-1] != 2:
        raise ValueError("states must end in [semantic, structural]")
    s, t = states[..., 0], states[..., 1]
    near = s + t
    if task == "additive":
        far = 1.5 * (s + t)
    elif task == "gated":
        gamma = float(interaction_strength)
        far = (1.5 - gamma) * (s + t) + gamma * s * t
    else:
        raise ValueError(f"unknown task {task!r}")
    return torch.stack((near, far), dim=-1)


class TinyRouter(nn.Module):
    """Small nonlinear categorical router with node-supervised contributions."""

    def __init__(self, hidden_dim: int, *, seed: int) -> None:
        super().__init__()
        torch.manual_seed(int(seed))
        self.network = nn.Sequential(
            nn.Linear(2, int(hidden_dim), dtype=DTYPE),
            nn.Tanh(),
            nn.Linear(int(hidden_dim), 2, dtype=DTYPE),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.network(states.to(dtype=DTYPE))


def train_model(
    config: ExperimentConfig,
    *,
    task: str,
    seed: int,
) -> tuple[TinyRouter, dict[str, float | int]]:
    target = teacher_contributions(
        STATES,
        task=task,
        interaction_strength=config.interaction_strength,
    )
    model = TinyRouter(config.hidden_dim, seed=int(seed))
    optimiser = torch.optim.Adam(model.parameters(), lr=float(config.learning_rate))
    final_loss = float("inf")
    step = 0
    for step in range(1, int(config.max_steps) + 1):
        prediction = model(STATES)
        loss = torch.mean((prediction - target) ** 2)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        final_loss = float(loss.detach())
        if final_loss <= float(config.early_stop_mse):
            break
    model.eval()
    with torch.no_grad():
        error = torch.abs(model(STATES) - target)
    return model, {
        "steps": int(step),
        "mse": final_loss,
        "max_abs_error": float(error.max()),
    }


def _normalise(values: torch.Tensor) -> torch.Tensor:
    values = torch.abs(values)
    total = values.sum()
    if not bool(torch.isfinite(total)) or float(total) <= 0:
        raise ValueError("cannot normalise a null or non-finite response")
    return values / total


def measure_four_state_field(
    contributions: torch.Tensor,
    *,
    effect_floor: float = 1.0e-5,
) -> dict[str, Any]:
    """Measure marginal profiles and the finite interaction carrier field.

    Rows must be ordered as clean ``(+,+)``, semantic donor ``(-,+)``,
    structural donor ``(+,-)``, and joint donor ``(-,-)``.
    """

    if tuple(contributions.shape) != (4, 2):
        raise ValueError("contributions must have shape [four states, two carriers]")
    clean, semantic, structural, joint = contributions
    semantic_response = clean - semantic
    structural_response = clean - structural
    interaction = clean - semantic - structural + joint
    semantic_profile = _normalise(semantic_response)
    structural_profile = _normalise(structural_response)
    interaction_mass = torch.abs(interaction)
    interaction_total = float(interaction_mass.sum())
    interaction_estimable = bool(
        np.isfinite(interaction_total) and interaction_total > float(effect_floor)
    )
    interaction_profile = (
        interaction_mass / interaction_mass.sum()
        if interaction_estimable
        else torch.full_like(interaction_mass, torch.nan)
    )
    return {
        "semantic_response": semantic_response.detach().cpu().numpy(),
        "structural_response": structural_response.detach().cpu().numpy(),
        "semantic_profile": semantic_profile.detach().cpu().numpy(),
        "structural_profile": structural_profile.detach().cpu().numpy(),
        "marginal_profile_tv": float(
            0.5 * torch.abs(semantic_profile - structural_profile).sum()
        ),
        "interaction": interaction.detach().cpu().numpy(),
        "output_interaction": float(interaction.sum()),
        "interaction_mass": interaction_total,
        "interaction_estimable": interaction_estimable,
        "interaction_profile": interaction_profile.detach().cpu().numpy(),
        "interaction_expected_distance": (
            float(np.dot(interaction_profile.detach().cpu().numpy(), DISTANCES))
            if interaction_estimable
            else float("nan")
        ),
        "far_interaction_share": (
            float(interaction_profile[-1]) if interaction_estimable else float("nan")
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
    """Train both tasks across seeds and return teacher/learned summaries."""

    config = ExperimentConfig() if config is None else config
    config.validate()
    torch.set_num_threads(1)
    result: dict[str, Any] = {"config": asdict(config), "tasks": {}}
    for task in ("additive", "gated"):
        teacher = teacher_contributions(
            STATES,
            task=task,
            interaction_strength=config.interaction_strength,
        )
        teacher_measurement = measure_four_state_field(
            teacher, effect_floor=config.interaction_effect_floor
        )
        seeds: list[dict[str, Any]] = []
        for seed in config.seeds:
            model, training = train_model(config, task=task, seed=int(seed))
            with torch.no_grad():
                learned = model(STATES)
            seeds.append(
                {
                    "seed": int(seed),
                    "training": training,
                    "measurement": measure_four_state_field(
                        learned, effect_floor=config.interaction_effect_floor
                    ),
                }
            )
        estimable_far_shares = [
            item["measurement"]["far_interaction_share"]
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
                "estimable_interaction_seeds": len(estimable_far_shares),
                "mean_far_interaction_share": (
                    float(np.mean(estimable_far_shares))
                    if estimable_far_shares
                    else None
                ),
                "max_marginal_profile_tv": float(
                    max(item["measurement"]["marginal_profile_tv"] for item in seeds)
                ),
            },
        }
    additive_profile = np.asarray(
        result["tasks"]["additive"]["teacher"]["semantic_profile"]
    )
    gated_profile = np.asarray(
        result["tasks"]["gated"]["teacher"]["semantic_profile"]
    )
    result["teacher_marginal_profile_tv_between_tasks"] = float(
        0.5 * np.abs(additive_profile - gated_profile).sum()
    )
    return _serialise(result)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(
        description="Run the tiny structure-gated conditional-carriage experiment."
    )
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--steps", type=int, default=4_000)
    parser.add_argument("--hidden-dim", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=3.0e-2)
    parser.add_argument("--interaction-strength", type=float, default=0.75)
    args = parser.parse_args(argv)
    config = ExperimentConfig(
        seeds=tuple(int(value) for value in args.seeds.split(",") if value),
        hidden_dim=int(args.hidden_dim),
        learning_rate=float(args.learning_rate),
        max_steps=int(args.steps),
        interaction_strength=float(args.interaction_strength),
    )
    result = run(config)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
