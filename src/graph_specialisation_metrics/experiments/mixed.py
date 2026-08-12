"""Mixed semantic–structural synthetic task."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from ..interventions import structural_donor_swap
from ..runner import clean_output_gradients, compute_channel_score
from ..sampling import sample_structural_donors
from . import ExperimentSetupError, save_scores

SEMANTIC = 0
STRUCTURAL = 1


@dataclass(frozen=True)
class Settings:
    nodes: int
    key_vocab: int
    classes: int
    rrwp_steps: int
    layers: int
    heads: int
    hidden_dim: int
    dropout: float
    attention_dropout: float
    seeds: tuple[int, ...]
    batch_size: int
    steps: int
    learning_rate: float
    weight_decay: float
    validation_graphs: int
    validate_every: int
    score_graphs: int
    donor_swaps_per_source: int
    score_seed: int
    early_stop_checks: int = 10
    early_stop_accuracy: float = 0.995
    acceptance_accuracy: float = 0.90

    @property
    def feature_dim(self) -> int:
        return 2 * self.key_vocab + self.classes + 4


def _settings(config: dict[str, Any], fast: bool) -> Settings:
    data = config.get("data", {})
    model = config.get("model", {})
    training = config.get("training", {})
    scoring = config.get("scoring", {})
    values = Settings(
        nodes=int(data.get("nodes", 16)),
        key_vocab=int(data.get("key_vocab", 32)),
        classes=int(data.get("classes", 8)),
        rrwp_steps=int(data.get("rrwp_steps", 10)),
        layers=int(model.get("layers", 3)),
        heads=int(model.get("heads", 8)),
        hidden_dim=int(model.get("hidden_dim", 96)),
        dropout=float(model.get("dropout", 0.0)),
        attention_dropout=float(model.get("attention_dropout", 0.05)),
        seeds=tuple(int(seed) for seed in training.get("seeds", (0, 1, 2))),
        batch_size=int(training.get("batch_size", 64)),
        steps=int(training.get("steps", 3000)),
        learning_rate=float(training.get("learning_rate", 2e-3)),
        weight_decay=float(training.get("weight_decay", 1e-5)),
        validation_graphs=int(training.get("validation_graphs", 512)),
        validate_every=int(training.get("validate_every", 100)),
        score_graphs=int(scoring.get("graphs", 96)),
        donor_swaps_per_source=int(scoring.get("donor_swaps_per_source", 6)),
        score_seed=int(scoring.get("seed", 31415)),
        early_stop_checks=int(training.get("early_stop_checks", 10)),
        early_stop_accuracy=float(training.get("early_stop_accuracy", 0.995)),
        acceptance_accuracy=float(training.get("acceptance_accuracy", 0.90)),
    )
    if fast:
        values = Settings(
            **{
                **asdict(values),
                "seeds": values.seeds[:1],
                "batch_size": min(values.batch_size, 32),
                "steps": min(values.steps, 300),
                "validation_graphs": min(values.validation_graphs, 64),
                "validate_every": min(values.validate_every, 100),
                "score_graphs": min(values.score_graphs, 4),
                "donor_swaps_per_source": min(values.donor_swaps_per_source, 2),
                "acceptance_accuracy": 0.0,
            }
        )
    if values.nodes % 2 or values.classes != values.nodes // 2:
        raise ValueError("mixed task requires an even node count and classes == nodes / 2")
    if values.key_vocab < values.nodes:
        raise ValueError("key_vocab must be at least the number of nodes")
    if values.hidden_dim % values.heads:
        raise ValueError("hidden_dim must be divisible by heads")
    if not values.seeds or values.donor_swaps_per_source < 1 or values.score_graphs < 1:
        raise ValueError("at least one training seed and one donor-swap are required")
    if values.early_stop_checks < 1 or values.validate_every < 1:
        raise ValueError("validation intervals must be positive")
    if not 0 <= values.acceptance_accuracy <= values.early_stop_accuracy <= 1:
        raise ValueError("training accuracy thresholds must lie in [0, 1]")
    return values


@dataclass
class Batch:
    x: Any
    rrwp: Any
    query: Any
    source: Any
    target: Any
    mode: Any

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def to(self, device: Any) -> Batch:
        return Batch(**{name: getattr(self, name).to(device) for name in self.__dataclass_fields__})

    def slice(self, start: int, stop: int) -> Batch:
        return Batch(
            **{name: getattr(self, name)[start:stop] for name in self.__dataclass_fields__}
        )


def _cycle_distance(nodes: int, left: int, right: int) -> int:
    delta = abs(int(left) - int(right))
    return min(delta, int(nodes) - delta)


@lru_cache(maxsize=8)
def _cycle_rrwp(nodes: int, steps: int) -> np.ndarray:
    adjacency = np.zeros((nodes, nodes), dtype=np.float32)
    index = np.arange(nodes)
    adjacency[index, (index + 1) % nodes] = 1
    adjacency[index, (index - 1) % nodes] = 1
    transition = adjacency / adjacency.sum(axis=1, keepdims=True)
    result = np.empty((nodes, nodes, steps), dtype=np.float32)
    power = np.eye(nodes, dtype=np.float32)
    for step in range(steps):
        if step:
            power = power @ transition
        result[:, :, step] = power
    return result


def _make_batch(settings: Settings, size: int, seed: int, mode: int | None = None) -> Batch:
    import torch

    rng = np.random.default_rng(int(seed))
    x = np.zeros((size, settings.nodes, settings.feature_dim), dtype=np.float32)
    rrwp = np.repeat(_cycle_rrwp(settings.nodes, settings.rrwp_steps)[None], size, axis=0)
    query = np.empty(size, dtype=np.int64)
    source = np.empty(size, dtype=np.int64)
    target = np.empty(size, dtype=np.int64)
    modes = np.arange(size, dtype=np.int64) % 2 if mode is None else np.full(size, mode)
    if mode is None:
        rng.shuffle(modes)

    value_start = settings.key_vocab
    query_key_start = settings.key_vocab + settings.classes
    query_flag = 2 * settings.key_vocab + settings.classes
    source_flag, semantic_flag, structural_flag = query_flag + 1, query_flag + 2, query_flag + 3
    for graph in range(size):
        keys = rng.permutation(settings.key_vocab)[: settings.nodes]
        values = rng.integers(0, settings.classes, size=settings.nodes)
        x[graph, np.arange(settings.nodes), keys] = 1
        x[graph, np.arange(settings.nodes), value_start + values] = 1
        q = int(rng.integers(0, settings.nodes))
        query[graph] = q
        x[graph, q, query_flag] = 1
        s = int(rng.choice([node for node in range(settings.nodes) if node != q]))
        if int(modes[graph]) == SEMANTIC:
            x[graph, q, query_key_start + keys[s]] = 1
            x[graph, q, semantic_flag] = 1
            target[graph] = int(values[s])
        else:
            x[graph, q, structural_flag] = 1
            target[graph] = _cycle_distance(settings.nodes, q, s) - 1
        source[graph] = s
        x[graph, s, source_flag] = 1
    return Batch(
        x=torch.from_numpy(x),
        rrwp=torch.from_numpy(rrwp),
        query=torch.from_numpy(query),
        source=torch.from_numpy(source),
        target=torch.from_numpy(target),
        mode=torch.from_numpy(modes.astype(np.int64)),
    )


def _model(settings: Settings):
    import torch
    from torch import nn

    try:
        from torch_geometric.data import Data

        from ._grit import grit_transformer_layer, layer_config

        GritTransformerLayer = grit_transformer_layer()
    except ImportError as exc:
        raise ExperimentSetupError(
            "Mixed-task dependencies are missing. Install this project with the mixed extra."
        ) from exc
    grit_config = layer_config(signed_sqrt=True)

    class MixedGRIT(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input = nn.Linear(settings.feature_dim, settings.hidden_dim)
            self.node_rrwp = nn.Linear(settings.rrwp_steps, settings.hidden_dim, bias=False)
            self.pair_rrwp = nn.Linear(settings.rrwp_steps, settings.hidden_dim, bias=False)
            self.edge_type = nn.Embedding(2, settings.hidden_dim)
            self.layers = nn.ModuleList(
                GritTransformerLayer(
                    settings.hidden_dim,
                    settings.hidden_dim,
                    settings.heads,
                    dropout=settings.dropout,
                    attn_dropout=settings.attention_dropout,
                    layer_norm=False,
                    batch_norm=True,
                    residual=True,
                    act="relu",
                    norm_e=True,
                    O_e=True,
                    cfg=grit_config,
                )
                for _ in range(settings.layers)
            )
            self.output = nn.Linear(settings.hidden_dim, settings.classes)

        def _batch(self, batch: Batch):
            graphs, nodes = len(batch), settings.nodes
            device = batch.x.device
            local = torch.arange(nodes, device=device)
            source = local.repeat_interleave(nodes)
            target = local.repeat(nodes)
            edge_source = torch.cat([source + graph * nodes for graph in range(graphs)])
            edge_target = torch.cat([target + graph * nodes for graph in range(graphs)])
            diagonal = torch.arange(nodes, device=device)
            node_rrwp = batch.rrwp[:, diagonal, diagonal].reshape(
                graphs * nodes, settings.rrwp_steps
            )
            data = Data(num_nodes=graphs * nodes)
            data.x = self.input(batch.x.reshape(graphs * nodes, -1)) + self.node_rrwp(node_rrwp)
            data.edge_index = torch.stack((edge_source, edge_target))
            edge_kind = (source != target).long().repeat(graphs)
            data.edge_attr = self.edge_type(edge_kind) + self.pair_rrwp(
                batch.rrwp.reshape(graphs * nodes * nodes, settings.rrwp_steps)
            )
            data.batch = torch.arange(graphs, device=device).repeat_interleave(nodes)
            # GRIT scales by degree in the input graph, not by dense attention support.
            data.deg = torch.full((graphs * nodes,), 2.0, device=device)
            data.log_deg = torch.log1p(data.deg)
            return data

        def forward(self, batch: Batch, *, capture: bool = False):
            data = self._batch(batch)
            head_outputs = []
            handles = []

            def capture_head(_module: Any, _inputs: Any, output: Any):
                head_value, edge_value = output
                value = head_value.reshape(
                    len(batch),
                    settings.nodes,
                    settings.heads,
                    settings.hidden_dim // settings.heads,
                ).clone()
                head_outputs.append(value)
                # Keep the recorded head output on the model-output path.
                return value.reshape_as(head_value), edge_value

            if capture:
                handles = [
                    layer.attention.register_forward_hook(capture_head) for layer in self.layers
                ]
            try:
                for layer in self.layers:
                    data = layer(data)
            finally:
                for handle in handles:
                    handle.remove()
            node_outputs = data.x.reshape(len(batch), settings.nodes, settings.hidden_dim)
            rows = torch.arange(len(batch), device=node_outputs.device)
            logits = self.output(node_outputs[rows, batch.query.long()])
            if not capture:
                return logits
            if len(head_outputs) != settings.layers:
                raise RuntimeError("a GRIT head-output hook did not fire")
            return logits, tuple(head_outputs)

    return MixedGRIT()


def _seed(value: int) -> None:
    import torch

    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def _device(config: dict[str, Any]):
    import torch

    requested = str(config.get("training", {}).get("device", "cuda"))
    return torch.device(
        requested if requested.startswith("cuda") and torch.cuda.is_available() else "cpu"
    )


def _batch_metrics(logits: Any, batch: Batch) -> dict[str, float]:
    import torch.nn.functional as F

    losses = F.cross_entropy(logits, batch.target.long(), reduction="none")
    correct = logits.argmax(dim=-1) == batch.target
    metrics = {
        "loss": float(losses.mean().detach().cpu()),
        "accuracy": float(correct.float().mean().detach().cpu()),
    }
    for mode, name in ((SEMANTIC, "semantic"), (STRUCTURAL, "structural")):
        mask = batch.mode == mode
        metrics[f"{name}_accuracy"] = float(correct[mask].float().mean().detach().cpu())
    return metrics


def train(config: dict[str, Any], *, output_dir: Path, fast: bool = False) -> dict[str, Any]:
    """Train every configured seed and return the checkpoint paths."""

    import torch
    import torch.nn.functional as F

    settings = _settings(config, fast)
    output_dir, device = Path(output_dir), _device(config)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for seed in settings.seeds:
        _seed(seed)
        model = _model(settings).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
        )
        scheduler = (
            torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
            if fast
            else torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=settings.steps, eta_min=settings.learning_rate * 0.05
            )
        )
        validation = _make_batch(settings, settings.validation_graphs, 50_000 + seed).to(device)
        best_loss = float("inf")
        best_state = None
        best_metrics: dict[str, float] = {}
        good_checks = 0
        model.train()
        for step in range(1, settings.steps + 1):
            batch = _make_batch(settings, settings.batch_size, seed * 1_000_003 + step).to(device)
            loss = F.cross_entropy(model(batch), batch.target.long())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            if step == 1 or step % settings.validate_every == 0 or step == settings.steps:
                model.eval()
                with torch.no_grad():
                    metrics = _batch_metrics(model(validation), validation)
                if metrics["loss"] < best_loss:
                    best_loss = metrics["loss"]
                    best_metrics = metrics
                    best_state = {
                        name: value.detach().cpu().clone()
                        for name, value in model.state_dict().items()
                    }
                if min(metrics["semantic_accuracy"], metrics["structural_accuracy"]) >= (
                    settings.early_stop_accuracy
                ):
                    good_checks += 1
                else:
                    good_checks = 0
                print(
                    f"mixed seed={seed} step={step}/{settings.steps} "
                    f"semantic={metrics['semantic_accuracy']:.3f} "
                    f"structural={metrics['structural_accuracy']:.3f}"
                )
                if good_checks >= settings.early_stop_checks:
                    break
                model.train()
        if best_state is None:
            raise RuntimeError("mixed training produced no checkpoint candidate")
        model.load_state_dict(best_state)
        model.eval()
        heldout = _make_batch(settings, settings.validation_graphs, 80_000 + seed).to(device)
        with torch.no_grad():
            heldout_metrics = _batch_metrics(model(heldout), heldout)
        if (
            min(heldout_metrics["semantic_accuracy"], heldout_metrics["structural_accuracy"])
            < settings.acceptance_accuracy
        ):
            raise RuntimeError(
                f"mixed seed {seed} failed the held-out {settings.acceptance_accuracy:.0%} "
                "per-task accuracy requirement"
            )
        path = checkpoint_dir / f"seed_{seed}.pt"
        torch.save(
            {
                "settings": asdict(settings),
                "seed": seed,
                "state_dict": best_state,
                "validation": best_metrics,
                "heldout": heldout_metrics,
            },
            path,
        )
        paths.append(path)
    return {"checkpoints": paths}


def _semantic_donor_swaps(settings: Settings, clean: Batch, count: int, rng: np.random.Generator):
    donor_swaps = []
    source = int(clean.source[0])
    value_slice = slice(settings.key_vocab, settings.key_vocab + settings.classes)
    current = int(clean.x[0, source, value_slice].argmax())
    for _ in range(count):
        value = int(rng.integers(0, settings.classes - 1))
        value += int(value >= current)
        donor_swap = clean.slice(0, 1)
        donor_swap.x = donor_swap.x.clone()
        donor_swap.x[0, source, value_slice] = 0
        donor_swap.x[0, source, settings.key_vocab + value] = 1
        donor_swaps.append(donor_swap)
    return donor_swaps


def _structural_donor_swaps(settings: Settings, clean: Batch, count: int, rng: np.random.Generator):
    source = int(clean.source[0])
    base = clean.rrwp[0]
    structural_profiles = [(base[node], base[:, node]) for node in range(settings.nodes)]
    selected = sample_structural_donors(structural_profiles, source, count, rng)
    donor_swaps = []
    for donor in selected:
        donor_swap = clean.slice(0, 1)
        _, pair_fields = structural_donor_swap({}, {"rrwp": base}, source, int(donor))
        donor_swap.rrwp[0] = pair_fields["rrwp"]
        donor_swaps.append(donor_swap)
    return donor_swaps


def _score_graph(model: Any, settings: Settings, clean: Batch, channel: str, rng: Any, device: Any):
    import torch

    clean = clean.to(device)
    logits, clean_head_outputs = model(clean, capture=True)
    clean_gradients = clean_output_gradients(logits[0], clean_head_outputs)[:, :, 0]

    cpu_clean = clean.to("cpu")
    if channel == "semantic":
        donor_swaps = _semantic_donor_swaps(
            settings, cpu_clean, settings.donor_swaps_per_source, rng
        )
    else:
        donor_swaps = _structural_donor_swaps(
            settings, cpu_clean, settings.donor_swaps_per_source, rng
        )
    donor_swap_head_outputs = []
    with torch.no_grad():
        for donor_swap in donor_swaps:
            _, head_outputs = model(donor_swap.to(device), capture=True)
            donor_swap_head_outputs.append(torch.stack([value[0] for value in head_outputs], dim=0))
    head_output_row_distances = np.asarray(
        [
            _cycle_distance(settings.nodes, int(clean.source[0]), node)
            for node in range(settings.nodes)
        ]
    )
    channel_score = compute_channel_score(
        torch.stack([value[0] for value in clean_head_outputs], dim=0),
        torch.stack(donor_swap_head_outputs, dim=0),
        clean_gradients,
        [0] * len(donor_swaps),
        [int(clean.source[0])] * len(donor_swaps),
        head_output_row_distances=head_output_row_distances,
    )
    contributions = channel_score.distance_resolved_score_contributions
    if contributions is None:  # pragma: no cover - distance categories are always supplied
        raise RuntimeError("mixed score has no distance-resolved score contributions")
    return channel_score.score, contributions.score_contributions


def _checkpoint_paths(checkpoint: Path | None, output_dir: Path) -> list[Path]:
    target = Path(checkpoint) if checkpoint is not None else output_dir / "checkpoints"
    paths = [target] if target.is_file() else sorted(target.glob("seed_*.pt"))
    if not paths:
        raise ExperimentSetupError(
            f"no mixed-task checkpoint found at {target}; run training first"
        )
    return paths


_MODEL_TASK_FIELDS = (
    "nodes",
    "key_vocab",
    "classes",
    "rrwp_steps",
    "layers",
    "heads",
    "hidden_dim",
    "dropout",
    "attention_dropout",
)


def _validate_checkpoint_settings(requested: Settings, saved: Settings) -> None:
    mismatches = [
        name for name in _MODEL_TASK_FIELDS if getattr(requested, name) != getattr(saved, name)
    ]
    if mismatches:
        raise ExperimentSetupError(
            "The mixed checkpoint does not match the current task/model config: "
            + ", ".join(mismatches)
        )


def score(
    config: dict[str, Any],
    *,
    checkpoint: Path | None,
    output_dir: Path,
    fast: bool = False,
) -> Path:
    """Compute specialisation scores and distance-resolved score contributions."""

    import torch

    output_dir, device = Path(output_dir), _device(config)
    requested = _settings(config, fast)
    semantic_all, structural_all = [], []
    semantic_distance_all, structural_distance_all = [], []
    seed_rows, layer_rows, head_rows = [], [], []
    records = []
    seen_seeds: set[int] = set()
    for path in _checkpoint_paths(checkpoint, output_dir):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        try:
            saved = Settings(**payload["settings"])
            seed = int(payload["seed"])
        except (KeyError, TypeError) as exc:
            raise ExperimentSetupError(f"invalid mixed checkpoint: {path}") from exc
        _validate_checkpoint_settings(requested, saved)
        if seed in seen_seeds:
            raise ExperimentSetupError(f"duplicate mixed checkpoint seed {seed}")
        seen_seeds.add(seed)
        records.append((payload, saved, seed))

    for payload, saved, seed in records:
        model = _model(saved)
        model.load_state_dict(payload["state_dict"])
        model.to(device).eval()
        # Evaluated graphs and donor-swaps are fixed across independently trained models.
        rng = np.random.default_rng(requested.score_seed + 2)
        sem_graphs, str_graphs, sem_distance, str_distance = [], [], [], []
        for graph in range(requested.score_graphs):
            sem_clean = _make_batch(requested, 1, requested.score_seed + graph, SEMANTIC)
            str_clean = _make_batch(requested, 1, requested.score_seed + graph, STRUCTURAL)
            score_value, distance_value = _score_graph(
                model, requested, sem_clean, "semantic", rng, device
            )
            sem_graphs.append(score_value)
            sem_distance.append(distance_value)
            score_value, distance_value = _score_graph(
                model, requested, str_clean, "structural", rng, device
            )
            str_graphs.append(score_value)
            str_distance.append(distance_value)
        semantic = np.mean(sem_graphs, axis=0)
        structural = np.mean(str_graphs, axis=0)
        semantic_distance = np.mean(sem_distance, axis=0)
        structural_distance = np.mean(str_distance, axis=0)
        semantic_all.append(semantic.reshape(-1))
        structural_all.append(structural.reshape(-1))
        semantic_distance_all.append(semantic_distance.reshape(-1, semantic_distance.shape[-1]))
        structural_distance_all.append(
            structural_distance.reshape(-1, structural_distance.shape[-1])
        )
        for layer in range(requested.layers):
            for head in range(requested.heads):
                seed_rows.append(seed)
                layer_rows.append(layer)
                head_rows.append(head)

    return save_scores(
        output_dir,
        np.concatenate(semantic_all),
        np.concatenate(structural_all),
        semantic_distance_contributions=np.concatenate(semantic_distance_all),
        structural_distance_contributions=np.concatenate(structural_distance_all),
        distance_categories=np.arange(semantic_distance_all[0].shape[-1]),
        seed=np.asarray(seed_rows),
        layer=np.asarray(layer_rows),
        head=np.asarray(head_rows),
    )
