"""Emergent redundant routing in a tiny multi-head graph transformer.

Five registered graph carriers are presented as tokens: one local anchor and
four distant records on real ZINC molecular supports at distances 1, 4, 4, and
6.  Semantic context selects a record key and structural context selects its
bank.  The selected value is also copied to the local anchor.

Unlike the controlled-mixture benchmark, this model has no explicit
local/global gate.  A two-layer, four-head transformer and graph-style summed
token readout must learn how much to use each route.  Route reliability is
manipulated only in the training data.  Clean final-state conditional carriage,
attention responses, and route-specific corruption performance are then
measured on the same held-out examples.  The default graph readout sums token
contributions so final-state carriage retains a non-degenerate distance field.
An anchor-only readout is retained as a sensitivity control: it removes a
readout bypass but necessarily collapses final-state carriage onto distance 0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .molecular_redundant_record_router import (
    Config as RouterConfig,
)
from .molecular_redundant_record_router import (
    RouterDataset,
    load_molecular_supports,
    make_router_dataset,
)
from .molecular_redundant_route_robustness import (
    corrupt_local_copy,
    corrupt_selected_distant_record,
)

DTYPE = torch.float32
TOKEN_DISTANCES = torch.tensor((0, 1, 4, 4, 6), dtype=torch.long)
DISTANCE_CATEGORIES = torch.tensor((0, 1, 4, 6), dtype=torch.long)


@dataclass(frozen=True)
class Config:
    output_dir: Path
    data_root: Path
    train_graphs: int = 64
    val_graphs: int = 16
    test_graphs: int = 16
    examples_per_graph: int = 8
    seeds: tuple[int, ...] = (0, 1, 2, 3)
    train_local_corruption: tuple[float, ...] = (0.0, 0.01, 0.05, 0.20)
    train_distant_corruption: float = 0.05
    hidden_dim: int = 32
    layers: int = 2
    heads: int = 4
    readout: str = "sum"
    max_steps: int = 2_000
    learning_rate: float = 3.0e-3
    weight_decay: float = 1.0e-4
    early_stop_patience: int = 250
    data_seed: int = 80_203

    def validate(self) -> None:
        if min(self.train_graphs, self.val_graphs, self.test_graphs) < 1:
            raise ValueError("each split requires at least one graph")
        if self.examples_per_graph < 4 or self.max_steps < 1:
            raise ValueError("invalid data or training size")
        if not self.seeds or len(self.seeds) != len(set(self.seeds)):
            raise ValueError("seeds must be non-empty and unique")
        if self.hidden_dim < 4 or self.hidden_dim % self.heads:
            raise ValueError("hidden_dim must be positive and divisible by heads")
        if self.layers < 1 or self.heads < 1 or self.early_stop_patience < 1:
            raise ValueError("layers, heads, and patience must be positive")
        if self.readout not in {"anchor", "sum"}:
            raise ValueError("readout must be 'anchor' or 'sum'")
        corruption = self.train_local_corruption
        if (
            not corruption
            or tuple(sorted(set(corruption))) != corruption
            or any(not 0 <= value < 0.5 for value in corruption)
        ):
            raise ValueError("local corruption must be unique, increasing, and in [0,.5)")
        if not 0 <= self.train_distant_corruption < 0.5:
            raise ValueError("distant corruption must lie in [0,.5)")


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, heads: int) -> None:
        super().__init__()
        self.norm_attention = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            heads,
            dropout=0.0,
            batch_first=True,
        )
        self.norm_feedforward = nn.LayerNorm(hidden_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalised = self.norm_attention(tokens)
        attended, weights = self.attention(
            normalised,
            normalised,
            normalised,
            need_weights=True,
            average_attn_weights=False,
        )
        tokens = tokens + attended
        tokens = tokens + self.feedforward(self.norm_feedforward(tokens))
        return tokens, weights


class TinyMultiHeadGraphTransformer(nn.Module):
    """Two-layer dense transformer with an anchor or summed graph readout."""

    feature_dim = 15

    def __init__(self, config: Config, *, seed: int) -> None:
        super().__init__()
        torch.manual_seed(int(seed))
        self.input_projection = nn.Linear(self.feature_dim, int(config.hidden_dim))
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(int(config.hidden_dim), int(config.heads))
                for _ in range(int(config.layers))
            ]
        )
        self.final_norm = nn.LayerNorm(int(config.hidden_dim))
        self.token_readout = nn.Linear(int(config.hidden_dim), 1)
        self.readout = str(config.readout)

    @staticmethod
    def token_features(
        dataset: RouterDataset,
        *,
        query: torch.Tensor | None = None,
        structure: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query = dataset.query if query is None else query
        structure = dataset.structure if structure is None else structure
        batch = len(dataset)
        device = dataset.record_values.device
        features = torch.zeros(batch, 5, 15, dtype=DTYPE, device=device)
        # Role: [local/query, record].
        features[:, 0, 0] = 1.0
        features[:, 1:, 1] = 1.0
        # Query and structural context are present only at the anchor.
        features[torch.arange(batch), 0, 2 + query] = 1.0
        features[torch.arange(batch), 0, 4 + structure] = 1.0
        # Record bank and key fields.
        banks = torch.tensor((0, 0, 1, 1), dtype=torch.long, device=device)
        keys = torch.tensor((0, 1, 0, 1), dtype=torch.long, device=device)
        record_rows = torch.arange(batch, device=device)[:, None]
        record_tokens = torch.arange(1, 5, device=device)[None, :]
        features[record_rows, record_tokens, 6 + banks[None, :]] = 1.0
        features[record_rows, record_tokens, 8 + keys[None, :]] = 1.0
        # Scalar semantic payload.
        features[:, 0, 10] = dataset.local_value.to(DTYPE)
        features[:, 1:, 10] = dataset.record_values.to(DTYPE)
        # Exact shortest-path category from the molecular support.
        for category_index, distance in enumerate(DISTANCE_CATEGORIES.tolist()):
            features[:, TOKEN_DISTANCES == int(distance), 11 + category_index] = 1.0
        return features

    def forward(
        self,
        dataset: RouterDataset,
        *,
        query: torch.Tensor | None = None,
        structure: torch.Tensor | None = None,
        return_details: bool = False,
    ) -> Any:
        tokens = self.input_projection(
            self.token_features(dataset, query=query, structure=structure)
        )
        attention: list[torch.Tensor] = []
        for block in self.blocks:
            tokens, weights = block(tokens)
            attention.append(weights)
        final_tokens = self.final_norm(tokens)
        token_outputs = self.token_readout(final_tokens).squeeze(-1)
        if self.readout == "anchor":
            output = token_outputs[:, 0]
        else:
            output = token_outputs.sum(dim=-1)
        if return_details:
            return output, final_tokens, tuple(attention)
        return output


def _to_device(dataset: RouterDataset, device: torch.device) -> RouterDataset:
    return RouterDataset(
        graph_ids=dataset.graph_ids.to(device),
        query=dataset.query.to(device),
        structure=dataset.structure.to(device),
        record_values=dataset.record_values.to(device),
        local_value=dataset.local_value.to(device),
        target=dataset.target.to(device),
    )


def train_model(
    config: Config,
    train: RouterDataset,
    validation: RouterDataset,
    *,
    seed: int,
) -> tuple[TinyMultiHeadGraphTransformer, dict[str, float | int]]:
    device = torch.device("cpu")
    train = _to_device(train, device)
    validation = _to_device(validation, device)
    model = TinyMultiHeadGraphTransformer(config, seed=int(seed)).to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    best_validation = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    step = 0
    for step in range(1, int(config.max_steps) + 1):
        model.train()
        prediction = model(train)
        loss = torch.mean((prediction - train.target.to(DTYPE)) ** 2)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(
                torch.mean((model(validation) - validation.target.to(DTYPE)) ** 2)
            )
        if validation_loss < best_validation - 1.0e-10:
            best_validation = validation_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= int(config.early_stop_patience):
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prediction = model(validation)
    return model, {
        "steps": int(step),
        "validation_mae": float(
            torch.mean(torch.abs(prediction - validation.target.to(DTYPE)))
        ),
        "validation_mse": best_validation,
    }


def _profile(response: torch.Tensor) -> tuple[float, float, float]:
    mass = torch.abs(response)
    total = mass.sum(dim=-1)
    valid = total > 1.0e-8
    normalised = torch.zeros_like(mass)
    normalised[valid] = mass[valid] / total[valid, None]
    distances = TOKEN_DISTANCES.to(device=response.device, dtype=response.dtype)
    expected = torch.sum(normalised * distances, dim=-1)
    far = normalised[:, TOKEN_DISTANCES.to(response.device) >= 4].sum(dim=-1)
    return (
        float(total.mean().detach().cpu()),
        float(expected[valid].mean().detach().cpu()),
        float(far[valid].mean().detach().cpu()),
    )


def _record_attention_profile(weights: torch.Tensor) -> torch.Tensor:
    # [batch, head, query token, key token] -> anchor query over four records.
    record = weights[:, :, 0, 1:]
    return record / record.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)


def measure_internal(
    model: TinyMultiHeadGraphTransformer,
    dataset: RouterDataset,
) -> dict[str, Any]:
    dataset = _to_device(dataset, torch.device("cpu"))
    query_donor = 1 - dataset.query
    structure_donor = 1 - dataset.structure
    model.zero_grad(set_to_none=True)
    clean_output, clean_state, clean_attention = model(
        dataset, return_details=True
    )
    clean_gradient = torch.autograd.grad(clean_output.sum(), clean_state)[0].detach()
    with torch.no_grad():
        semantic_output, semantic_state, semantic_attention = model(
            dataset, query=query_donor, return_details=True
        )
        structural_output, structural_state, structural_attention = model(
            dataset, structure=structure_donor, return_details=True
        )
        joint_output, joint_state, joint_attention = model(
            dataset,
            query=query_donor,
            structure=structure_donor,
            return_details=True,
        )
        semantic_delta = clean_state.detach() - semantic_state
        structural_delta = clean_state.detach() - structural_state
        interaction_delta = (
            clean_state.detach() - semantic_state - structural_state + joint_state
        )
        semantic_carriage = torch.abs(torch.sum(semantic_delta * clean_gradient, dim=-1))
        structural_carriage = torch.abs(
            torch.sum(structural_delta * clean_gradient, dim=-1)
        )
        interaction_carriage = torch.abs(
            torch.sum(interaction_delta * clean_gradient, dim=-1)
        )

    semantic_mass, semantic_distance, semantic_far = _profile(semantic_carriage)
    structural_mass, structural_distance, structural_far = _profile(structural_carriage)
    interaction_mass, interaction_distance, interaction_far = _profile(
        interaction_carriage
    )
    layer_records: list[dict[str, Any]] = []
    for layer, (clean, semantic, structural, joint) in enumerate(
        zip(
            clean_attention,
            semantic_attention,
            structural_attention,
            joint_attention,
            strict=True,
        )
    ):
        clean_profile = _record_attention_profile(clean.detach())
        semantic_profile = _record_attention_profile(semantic)
        structural_profile = _record_attention_profile(structural)
        joint_profile = _record_attention_profile(joint)
        semantic_score = 0.5 * torch.abs(clean_profile - semantic_profile).sum(dim=-1)
        structural_score = 0.5 * torch.abs(clean_profile - structural_profile).sum(dim=-1)
        interaction_score = 0.5 * torch.abs(
            clean_profile - semantic_profile - structural_profile + joint_profile
        ).sum(dim=-1)
        selected = 2 * dataset.structure + dataset.query
        selected_attention = clean_profile.gather(
            2,
            selected[:, None, None].expand(-1, clean_profile.shape[1], 1),
        ).squeeze(-1)
        for head in range(clean_profile.shape[1]):
            layer_records.append(
                {
                    "layer": int(layer),
                    "head": int(head),
                    "semantic_score": float(semantic_score[:, head].mean()),
                    "structural_score": float(structural_score[:, head].mean()),
                    "attention_interaction_score": float(
                        interaction_score[:, head].mean()
                    ),
                    "selected_record_attention": float(
                        selected_attention[:, head].mean()
                    ),
                }
            )
    semantic_scores = np.asarray(
        [record["semantic_score"] for record in layer_records], dtype=np.float64
    )
    structural_scores = np.asarray(
        [record["structural_score"] for record in layer_records], dtype=np.float64
    )
    attention_interactions = np.asarray(
        [record["attention_interaction_score"] for record in layer_records],
        dtype=np.float64,
    )
    head_score_alignment = float(
        np.dot(semantic_scores, structural_scores)
        / max(
            float(np.linalg.norm(semantic_scores) * np.linalg.norm(structural_scores)),
            1.0e-12,
        )
    )
    head_relative_imbalance = float(
        np.mean(
            np.abs(semantic_scores - structural_scores)
            / (semantic_scores + structural_scores + 1.0e-8)
        )
    )
    output_interaction = torch.abs(
        clean_output.detach() - semantic_output - structural_output + joint_output
    )
    return {
        "semantic_effect_mass": semantic_mass,
        "structural_effect_mass": structural_mass,
        "interaction_effect_mass": interaction_mass,
        "semantic_expected_distance": semantic_distance,
        "structural_expected_distance": structural_distance,
        "interaction_expected_distance": interaction_distance,
        "semantic_far_share": semantic_far,
        "structural_far_share": structural_far,
        "interaction_far_share": interaction_far,
        "mean_abs_output_interaction": float(output_interaction.mean()),
        "mean_head_semantic_score": float(semantic_scores.mean()),
        "mean_head_structural_score": float(structural_scores.mean()),
        "mean_head_attention_interaction": float(attention_interactions.mean()),
        "max_head_attention_interaction": float(attention_interactions.max()),
        "head_division_of_labour": float(np.mean(np.abs(semantic_scores - structural_scores))),
        "head_score_alignment": head_score_alignment,
        "head_relative_imbalance": head_relative_imbalance,
        "heads": layer_records,
    }


def measure_failures(
    model: TinyMultiHeadGraphTransformer,
    clean: RouterDataset,
) -> dict[str, float]:
    clean = _to_device(clean, torch.device("cpu"))
    local_failed = _to_device(
        corrupt_local_copy(clean, probability=1.0, seed=0), torch.device("cpu")
    )
    distant_failed = _to_device(
        corrupt_selected_distant_record(clean), torch.device("cpu")
    )
    with torch.no_grad():
        clean_mae = float(torch.mean(torch.abs(model(clean) - clean.target.to(DTYPE))))
        local_failure_mae = float(
            torch.mean(torch.abs(model(local_failed) - clean.target.to(DTYPE)))
        )
        distant_failure_mae = float(
            torch.mean(torch.abs(model(distant_failed) - clean.target.to(DTYPE)))
        )
        local_only_failure = float(
            torch.mean(torch.abs(local_failed.local_value - clean.target))
        )
    return {
        "clean_mae": clean_mae,
        "local_failure_mae": local_failure_mae,
        "distant_failure_mae": distant_failure_mae,
        "local_only_failure_mae": local_only_failure,
        "local_failure_rescue": local_only_failure - local_failure_mae,
        "distant_failure_damage": distant_failure_mae - clean_mae,
    }


def _load_datasets(config: Config) -> dict[str, RouterDataset]:
    router_config = RouterConfig(
        output_dir=config.output_dir,
        data_root=config.data_root,
        train_graphs=config.train_graphs,
        val_graphs=config.val_graphs,
        test_graphs=config.test_graphs,
        examples_per_graph=config.examples_per_graph,
        seeds=config.seeds,
        local_mixes=(0.5,),
        data_seed=config.data_seed,
    )
    split_specs = (
        ("train", router_config.train_graphs, router_config.data_seed + 1),
        ("val", router_config.val_graphs, router_config.data_seed + 2),
        ("test", router_config.test_graphs, router_config.data_seed + 3),
    )
    supports = {
        split: load_molecular_supports(
            router_config.data_root, split=split, count=count, seed=seed
        )
        for split, count, seed in split_specs
    }
    return {
        split: make_router_dataset(
            supports[split],
            examples_per_graph=router_config.examples_per_graph,
            seed=router_config.data_seed + 100 + index,
        )
        for index, (split, _, _) in enumerate(split_specs)
    }


def _mean(records: Sequence[Mapping[str, float]]) -> dict[str, float]:
    keys = [key for key, value in records[0].items() if isinstance(value, (int, float))]
    return {
        key: float(np.mean([float(record[key]) for record in records])) for key in keys
    }


def _correlation(records: Sequence[Mapping[str, float]], left: str, right: str) -> float:
    x = np.asarray([float(record[left]) for record in records], dtype=np.float64)
    y = np.asarray([float(record[right]) for record in records], dtype=np.float64)
    if x.size < 2 or float(np.std(x)) <= 1.0e-12 or float(np.std(y)) <= 1.0e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _serialise(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _serialise(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialise(item) for item in value]
    return value


def run(config: Config) -> dict[str, Any]:
    config.validate()
    torch.set_num_threads(1)
    datasets = _load_datasets(config)
    result: dict[str, Any] = {"config": asdict(config), "conditions": {}}
    flat: list[dict[str, Any]] = []
    for probability in config.train_local_corruption:
        seed_records: list[dict[str, Any]] = []
        for seed in config.seeds:
            train = corrupt_local_copy(
                datasets["train"],
                probability=float(probability),
                seed=int(config.data_seed + 10_000 * probability + seed),
            )
            validation = corrupt_local_copy(
                datasets["val"],
                probability=float(probability),
                seed=int(config.data_seed + 20_000 * probability + seed),
            )
            train = corrupt_selected_distant_record(
                train,
                probability=float(config.train_distant_corruption),
                seed=int(config.data_seed + 30_000 * probability + seed),
            )
            validation = corrupt_selected_distant_record(
                validation,
                probability=float(config.train_distant_corruption),
                seed=int(config.data_seed + 40_000 * probability + seed),
            )
            model, training = train_model(
                config, train, validation, seed=int(seed)
            )
            internal = measure_internal(model, datasets["test"])
            failures = measure_failures(model, datasets["test"])
            measurement = {**failures, **internal}
            seed_records.append(
                {
                    "seed": int(seed),
                    "training": training,
                    "measurement": measurement,
                }
            )
            flat.append(
                {
                    "train_local_corruption": float(probability),
                    "seed": int(seed),
                    **{key: value for key, value in measurement.items() if key != "heads"},
                }
            )
        result["conditions"][str(probability)] = {
            "train_local_corruption": float(probability),
            "train_distant_corruption": float(config.train_distant_corruption),
            "seeds": seed_records,
            "mean": _mean(
                [
                    {
                        key: value
                        for key, value in record["measurement"].items()
                        if key != "heads"
                    }
                    for record in seed_records
                ]
            ),
        }
    result["correlations"] = {
        "interaction_mass_vs_local_rescue": _correlation(
            flat, "interaction_effect_mass", "local_failure_rescue"
        ),
        "interaction_mass_vs_distant_damage": _correlation(
            flat, "interaction_effect_mass", "distant_failure_damage"
        ),
        "attention_interaction_vs_local_rescue": _correlation(
            flat, "mean_head_attention_interaction", "local_failure_rescue"
        ),
        "head_division_vs_local_rescue": _correlation(
            flat, "head_division_of_labour", "local_failure_rescue"
        ),
        "head_score_alignment_vs_interaction_mass": _correlation(
            flat, "head_score_alignment", "interaction_effect_mass"
        ),
        "head_relative_imbalance_vs_interaction_mass": _correlation(
            flat, "head_relative_imbalance", "interaction_effect_mass"
        ),
        "train_corruption_vs_interaction_mass": _correlation(
            flat, "train_local_corruption", "interaction_effect_mass"
        ),
    }
    payload = _serialise(result)
    fingerprint_record = {**payload["config"], "output_dir": None}
    payload["fingerprint"] = hashlib.sha256(
        json.dumps(fingerprint_record, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload


def _parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _parse_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/molecular_multihead_redundancy"),
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-graphs", type=int, default=64)
    parser.add_argument("--val-graphs", type=int, default=16)
    parser.add_argument("--test-graphs", type=int, default=16)
    parser.add_argument("--examples-per-graph", type=int, default=8)
    parser.add_argument("--seeds", default="0,1,2,3")
    parser.add_argument("--train-local-corruption", default="0,0.01,0.05,0.2")
    parser.add_argument("--train-distant-corruption", type=float, default=0.05)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--readout", choices=("anchor", "sum"), default="sum")
    parser.add_argument("--max-steps", type=int, default=2_000)
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    config = Config(
        output_dir=args.output_dir,
        data_root=args.data_root,
        train_graphs=int(args.train_graphs),
        val_graphs=int(args.val_graphs),
        test_graphs=int(args.test_graphs),
        examples_per_graph=int(args.examples_per_graph),
        seeds=_parse_ints(args.seeds),
        train_local_corruption=_parse_floats(args.train_local_corruption),
        train_distant_corruption=float(args.train_distant_corruption),
        hidden_dim=int(args.hidden_dim),
        layers=int(args.layers),
        heads=int(args.heads),
        readout=str(args.readout),
        max_steps=int(args.max_steps),
    )
    result = run(config)
    summary = {
        "conditions": {
            key: value["mean"] for key, value in result["conditions"].items()
        },
        "correlations": result["correlations"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":  # pragma: no cover
    main()
