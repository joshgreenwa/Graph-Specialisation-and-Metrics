"""Minimal order/identity dissociation experiment.

This module implements the experiment specified in
``minimal_order_dissociation_experiment.md``.  It deliberately keeps the data
generator, models, RQ3-style interventions, and poster figures in one compact
runner so the introductory evidence can be reproduced without touching the
larger teacher-student CFIM artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


TASKS = ("add", "rel")
MODELS = ("mpnn", "mpnn_vn", "gt", "nope_gt")
MODEL_LABELS = {
    "mpnn": "MPNN",
    "mpnn_vn": "MPNN+VN",
    "gt": "GT",
    "nope_gt": "NoPE-GT",
}
TASK_LABELS = {
    "add": "ADD",
    "rel": "REL",
}
DEFAULT_SEEDS = (1001, 1002, 1003)
EPS = 1.0e-12
INTERACTION_SIGNAL_EPS = 1.0e-4


@dataclass(frozen=True)
class ExperimentSpec:
    n_nodes: int = 32
    receptive_radius: int = 4
    alphabet_size: int = 4
    train_size: int = 5000
    val_size: int = 1000
    test_size: int = 1000
    data_seed: int = 7001

    @property
    def value_feature_dim(self) -> int:
        return self.alphabet_size + 1

    @property
    def label_add_range(self) -> float:
        # Values are distinct in {0, 1, 2, 3}; the attainable sum range is 1..5.
        return float((self.alphabet_size - 1) + (self.alphabet_size - 2) - (0 + 1))

    @property
    def position_low(self) -> int:
        return self.receptive_radius + 2

    @property
    def position_high(self) -> int:
        return self.n_nodes - 1


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    ensure_dir(path.parent)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_csv_list(text: str, *, allowed: Sequence[str] | None = None) -> list[str]:
    values = [value.strip() for value in text.split(",") if value.strip()]
    if not values:
        raise argparse.ArgumentTypeError("list cannot be empty")
    if allowed is not None:
        bad = [value for value in values if value not in allowed]
        if bad:
            raise argparse.ArgumentTypeError(f"unknown values {bad}; expected one of {list(allowed)}")
    return values


def parse_seed_list(text: str) -> list[int]:
    return [int(value.strip()) for value in text.split(",") if value.strip()]


def default_output_root() -> Path:
    env = Path.cwd() / "artifacts" / "minimal_order_dissociation"
    return env


def data_path(root: Path, spec: ExperimentSpec) -> Path:
    return root / "data" / f"path_order_N{spec.n_nodes}_r{spec.receptive_radius}_seed{spec.data_seed}.pt"


def checkpoint_path(root: Path, task: str, model_name: str, seed: int) -> Path:
    return root / "checkpoints" / task / model_name / f"seed_{int(seed)}" / "best.pt"


def clean_metrics_path(root: Path) -> Path:
    return root / "metrics" / "clean_performance.csv"


def rq3_rows_path(root: Path) -> Path:
    return root / "metrics" / "rq3_interactions.csv"


def rq3_summary_path(root: Path) -> Path:
    return root / "metrics" / "rq3_summary.csv"


def config_record_path(root: Path) -> Path:
    return root / "metrics" / "configuration.json"


def figures_dir(root: Path) -> Path:
    return root / "figures"


def value_feature(value: int | None, alphabet_size: int) -> torch.Tensor:
    vec = torch.zeros(alphabet_size + 1, dtype=torch.float32)
    if value is None:
        vec[0] = 1.0
    else:
        vec[int(value) + 1] = 1.0
    return vec


def encode_values(
    p1: torch.Tensor,
    p2: torch.Tensor,
    x1: torch.Tensor,
    x2: torch.Tensor,
    spec: ExperimentSpec,
) -> torch.Tensor:
    batch_size = int(p1.numel())
    x = torch.zeros(batch_size, spec.n_nodes, spec.value_feature_dim, dtype=torch.float32, device=p1.device)
    x[:, :, 0] = 1.0
    rows = torch.arange(batch_size, device=p1.device)
    x[rows, p1.long(), 0] = 0.0
    x[rows, p2.long(), 0] = 0.0
    x[rows, p1.long(), x1.long() + 1] = 1.0
    x[rows, p2.long(), x2.long() + 1] = 1.0
    return x


def make_split(spec: ExperimentSpec, count: int, seed: int) -> dict[str, torch.Tensor]:
    rng = random.Random(int(seed))
    candidates = list(range(spec.position_low, spec.position_high + 1))
    alphabet = list(range(spec.alphabet_size))
    p1_rows: list[int] = []
    p2_rows: list[int] = []
    x1_rows: list[int] = []
    x2_rows: list[int] = []
    y_add_rows: list[float] = []
    y_rel_rows: list[int] = []
    for _ in range(int(count)):
        p1, p2 = sorted(rng.sample(candidates, 2))
        x1, x2 = rng.sample(alphabet, 2)
        p1_rows.append(p1)
        p2_rows.append(p2)
        x1_rows.append(x1)
        x2_rows.append(x2)
        y_add_rows.append(float(x1 + x2))
        y_rel_rows.append(int(x1 > x2))
    p1_t = torch.tensor(p1_rows, dtype=torch.long)
    p2_t = torch.tensor(p2_rows, dtype=torch.long)
    x1_t = torch.tensor(x1_rows, dtype=torch.long)
    x2_t = torch.tensor(x2_rows, dtype=torch.long)
    return {
        "x": encode_values(p1_t, p2_t, x1_t, x2_t, spec),
        "p1": p1_t,
        "p2": p2_t,
        "x1": x1_t,
        "x2": x2_t,
        "y_add": torch.tensor(y_add_rows, dtype=torch.float32),
        "y_rel": torch.tensor(y_rel_rows, dtype=torch.long),
    }


def build_dataset(spec: ExperimentSpec) -> dict[str, Any]:
    train = make_split(spec, spec.train_size, spec.data_seed + 101)
    val = make_split(spec, spec.val_size, spec.data_seed + 202)
    test = make_split(spec, spec.test_size, spec.data_seed + 303)
    validate_dataset({"splits": {"train": train, "val": val, "test": test}}, spec)
    return {
        "spec": asdict(spec),
        "splits": {"train": train, "val": val, "test": test},
    }


def validate_dataset(dataset: Mapping[str, Any], spec: ExperimentSpec) -> None:
    for split_name, split in dataset["splits"].items():
        p1 = split["p1"]
        p2 = split["p2"]
        x1 = split["x1"]
        x2 = split["x2"]
        if not bool(torch.all(p1 < p2)):
            raise ValueError(f"{split_name}: expected p1 < p2")
        if not bool(torch.all(p1 > spec.receptive_radius)):
            raise ValueError(f"{split_name}: p1 must be beyond receptive radius")
        if not bool(torch.all(p2 > spec.receptive_radius)):
            raise ValueError(f"{split_name}: p2 must be beyond receptive radius")
        if not bool(torch.all(x1 != x2)):
            raise ValueError(f"{split_name}: values must be distinct")
        expected_add = (x1 + x2).float()
        expected_rel = (x1 > x2).long()
        if not torch.allclose(split["y_add"], expected_add):
            raise ValueError(f"{split_name}: ADD labels are inconsistent")
        if not torch.equal(split["y_rel"], expected_rel):
            raise ValueError(f"{split_name}: REL labels are inconsistent")
        null_rows = split["x"][:, :, 0]
        value_mass = split["x"][:, :, 1:].sum(dim=-1)
        if not torch.allclose(null_rows + value_mass, torch.ones_like(null_rows)):
            raise ValueError(f"{split_name}: feature rows are not null-or-value one-hot")


def save_dataset(root: Path, spec: ExperimentSpec, *, overwrite: bool = False) -> Path:
    path = data_path(root, spec)
    if path.exists() and not overwrite:
        print(f"[data] using existing cache {path}")
        return path
    ensure_dir(path.parent)
    dataset = build_dataset(spec)
    torch.save(dataset, path)
    print(
        "[data] wrote "
        f"{path} train={spec.train_size} val={spec.val_size} test={spec.test_size} "
        f"N={spec.n_nodes} r={spec.receptive_radius}"
    )
    return path


def load_dataset(path: Path) -> tuple[ExperimentSpec, dict[str, dict[str, torch.Tensor]]]:
    payload = torch.load(path, map_location="cpu")
    spec = ExperimentSpec(**payload["spec"])
    splits = payload["splits"]
    validate_dataset({"splits": splits}, spec)
    return spec, splits


def path_adjacency(n_nodes: int, device: torch.device | None = None) -> torch.Tensor:
    adj = torch.zeros(n_nodes, n_nodes, dtype=torch.float32, device=device)
    idx = torch.arange(n_nodes - 1, device=device)
    adj[idx, idx + 1] = 1.0
    adj[idx + 1, idx] = 1.0
    return adj


def path_distances(n_nodes: int, device: torch.device | None = None) -> torch.Tensor:
    idx = torch.arange(n_nodes, device=device)
    return (idx[:, None] - idx[None, :]).abs().long()


class ResidualMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MPNNModel(nn.Module):
    """Depth-r local message passing, optionally with a symmetric virtual node."""

    def __init__(
        self,
        *,
        n_nodes: int,
        in_dim: int,
        hidden_dim: int,
        depth: int,
        out_dim: int,
        use_virtual_node: bool,
    ):
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.depth = int(depth)
        self.use_virtual_node = bool(use_virtual_node)
        self.encoder = nn.Linear(in_dim, hidden_dim)
        msg_in = hidden_dim * (3 if use_virtual_node else 2)
        self.layers = nn.ModuleList([ResidualMLP(msg_in, hidden_dim * 2, hidden_dim) for _ in range(depth)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(depth)])
        if use_virtual_node:
            self.vn_layers = nn.ModuleList([ResidualMLP(hidden_dim * 2, hidden_dim * 2, hidden_dim) for _ in range(depth)])
            self.vn_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(depth)])
            readout_dim = hidden_dim * 2
        else:
            self.vn_layers = nn.ModuleList()
            self.vn_norms = nn.ModuleList()
            readout_dim = hidden_dim
        self.readout = nn.Sequential(
            nn.LayerNorm(readout_dim),
            nn.Linear(readout_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        adj = path_adjacency(n_nodes)
        deg = adj.sum(dim=-1).clamp_min(1.0)
        self.register_buffer("adj_norm", adj / deg[:, None], persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        vn = h.mean(dim=1) if self.use_virtual_node else None
        adj = self.adj_norm.to(dtype=h.dtype, device=h.device)
        for idx, layer in enumerate(self.layers):
            neigh = torch.einsum("ij,bjh->bih", adj, h)
            if self.use_virtual_node:
                assert vn is not None
                vn_broadcast = vn[:, None, :].expand_as(h)
                update = layer(torch.cat([h, neigh, vn_broadcast], dim=-1))
                h = self.norms[idx](h + update)
                vn_update = self.vn_layers[idx](torch.cat([vn, h.mean(dim=1)], dim=-1))
                vn = self.vn_norms[idx](vn + vn_update)
            else:
                update = layer(torch.cat([h, neigh], dim=-1))
                h = self.norms[idx](h + update)
        receiver = h[:, 0, :]
        if self.use_virtual_node:
            assert vn is not None
            receiver = torch.cat([receiver, vn], dim=-1)
        return self.readout(receiver)


class GraphTransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, *, n_nodes: int, use_distance_bias: bool):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.head_dim = hidden_dim // num_heads
        self.use_distance_bias = bool(use_distance_bias)
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.ln2 = nn.LayerNorm(hidden_dim)
        if use_distance_bias:
            self.distance_bias = nn.Embedding(n_nodes, num_heads)
        else:
            self.distance_bias = None
        self.register_buffer("dist", path_distances(n_nodes), persistent=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        bsz, n_nodes, _ = h.shape
        qkv = self.qkv(h).view(bsz, n_nodes, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = torch.einsum("bhid,bhjd->bhij", q, k) / math.sqrt(self.head_dim)
        if self.distance_bias is not None:
            bias = self.distance_bias(self.dist.to(h.device)).permute(2, 0, 1)
            scores = scores + bias[None, :, :, :].to(dtype=scores.dtype)
        attn = torch.softmax(scores, dim=-1)
        context = torch.einsum("bhij,bhjd->bhid", attn, v)
        context = context.transpose(1, 2).contiguous().view(bsz, n_nodes, self.hidden_dim)
        h = self.ln1(h + self.out(context))
        h = self.ln2(h + self.ffn(h))
        return h


class DenseGraphTransformer(nn.Module):
    """Dense GT with optional topology-derived position injection."""

    def __init__(
        self,
        *,
        n_nodes: int,
        in_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        out_dim: int,
        use_structural_pe: bool,
    ):
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.use_structural_pe = bool(use_structural_pe)
        self.encoder = nn.Linear(in_dim, hidden_dim)
        if use_structural_pe:
            self.position_embedding = nn.Embedding(n_nodes, hidden_dim)
        else:
            self.position_embedding = None
        self.blocks = nn.ModuleList(
            [
                GraphTransformerBlock(
                    hidden_dim,
                    num_heads,
                    n_nodes=n_nodes,
                    use_distance_bias=use_structural_pe,
                )
                for _ in range(num_layers)
            ]
        )
        self.readout = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.register_buffer("positions", torch.arange(n_nodes, dtype=torch.long), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        if self.position_embedding is not None:
            pos = self.position_embedding(self.positions.to(x.device))[None, :, :]
            h = h + pos.to(dtype=h.dtype)
        for block in self.blocks:
            h = block(h)
        return self.readout(h[:, 0, :])


def build_model(
    model_name: str,
    task: str,
    spec: ExperimentSpec,
    *,
    hidden_dim: int,
    gt_layers: int,
    gt_heads: int,
) -> nn.Module:
    out_dim = 1 if task == "add" else 2
    if model_name == "mpnn":
        return MPNNModel(
            n_nodes=spec.n_nodes,
            in_dim=spec.value_feature_dim,
            hidden_dim=hidden_dim,
            depth=spec.receptive_radius,
            out_dim=out_dim,
            use_virtual_node=False,
        )
    if model_name == "mpnn_vn":
        return MPNNModel(
            n_nodes=spec.n_nodes,
            in_dim=spec.value_feature_dim,
            hidden_dim=hidden_dim,
            depth=spec.receptive_radius,
            out_dim=out_dim,
            use_virtual_node=True,
        )
    if model_name in {"gt", "nope_gt"}:
        return DenseGraphTransformer(
            n_nodes=spec.n_nodes,
            in_dim=spec.value_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=gt_layers,
            num_heads=gt_heads,
            out_dim=out_dim,
            use_structural_pe=(model_name == "gt"),
        )
    raise ValueError(f"unknown model {model_name!r}")


def scalar_output(task: str, output: torch.Tensor) -> torch.Tensor:
    if task == "add":
        return output.reshape(-1)
    return output[:, 1] - output[:, 0]


def task_loss(task: str, output: torch.Tensor, split: Mapping[str, torch.Tensor]) -> torch.Tensor:
    if task == "add":
        return F.mse_loss(output.reshape(-1), split["y_add"].to(output.device))
    return F.cross_entropy(output, split["y_rel"].to(output.device))


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    task: str,
    split: Mapping[str, torch.Tensor],
    spec: ExperimentSpec,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    add_abs: list[torch.Tensor] = []
    rel_correct = 0
    rel_total = 0
    n = int(split["x"].shape[0])
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = {key: value[start:end].to(device) for key, value in split.items()}
        output = model(batch["x"])
        loss = task_loss(task, output, batch)
        losses.append(float(loss.item()) * (end - start))
        if task == "add":
            err = (output.reshape(-1) - batch["y_add"]).abs().detach().cpu()
            add_abs.append(err)
        else:
            pred = output.argmax(dim=-1)
            rel_correct += int((pred == batch["y_rel"]).sum().item())
            rel_total += int(end - start)
    mean_loss = sum(losses) / max(n, 1)
    if task == "add":
        abs_err = torch.cat(add_abs) if add_abs else torch.empty(0)
        mae = float(abs_err.mean().item()) if abs_err.numel() else float("nan")
        score = 1.0 - mae / max(spec.label_add_range, EPS)
        return {"loss": mean_loss, "mae": mae, "score": score, "accuracy": float("nan")}
    accuracy = rel_correct / max(rel_total, 1)
    return {"loss": mean_loss, "mae": float("nan"), "score": accuracy, "accuracy": accuracy}


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def batches(split: Mapping[str, torch.Tensor], batch_size: int, *, rng: torch.Generator) -> Iterable[dict[str, torch.Tensor]]:
    n = int(split["x"].shape[0])
    order = torch.randperm(n, generator=rng)
    for start in range(0, n, batch_size):
        idx = order[start : start + batch_size]
        yield {key: value[idx] for key, value in split.items()}


def train_one(
    *,
    root: Path,
    data_file: Path,
    task: str,
    model_name: str,
    seed: int,
    device: torch.device,
    hidden_dim: int,
    gt_layers: int,
    gt_heads: int,
    batch_size: int,
    max_epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    log_every: int,
    overwrite: bool,
) -> Path:
    ckpt = checkpoint_path(root, task, model_name, seed)
    if ckpt.exists() and not overwrite:
        print(f"[train] using existing checkpoint task={task} model={model_name} seed={seed}: {ckpt}")
        return ckpt
    spec, splits = load_dataset(data_file)
    set_seed(seed)
    model = build_model(
        model_name,
        task,
        spec,
        hidden_dim=hidden_dim,
        gt_layers=gt_layers,
        gt_heads=gt_heads,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    rng = torch.Generator().manual_seed(int(seed) + 17)
    best_loss = float("inf")
    best_epoch = -1
    best_metrics: dict[str, float] = {}
    ensure_dir(ckpt.parent)
    print(
        f"[train] start task={task} model={model_name} seed={seed} "
        f"params={sum(p.numel() for p in model.parameters()):,} device={device}"
    )
    for epoch in range(1, max_epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for batch_cpu in batches(splits["train"], batch_size, rng=rng):
            batch = {key: value.to(device) for key, value in batch_cpu.items()}
            optimizer.zero_grad(set_to_none=True)
            output = model(batch["x"])
            loss = task_loss(task, output, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total += float(loss.item()) * int(batch["x"].shape[0])
            count += int(batch["x"].shape[0])
        val_metrics = evaluate_model(
            model,
            task,
            splits["val"],
            spec,
            device=device,
            batch_size=batch_size,
        )
        train_loss = total / max(count, 1)
        improved = val_metrics["loss"] < best_loss - 1.0e-7
        if improved:
            best_loss = val_metrics["loss"]
            best_epoch = epoch
            best_metrics = dict(val_metrics)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "task": task,
                    "model_name": model_name,
                    "seed": int(seed),
                    "spec": asdict(spec),
                    "hidden_dim": int(hidden_dim),
                    "gt_layers": int(gt_layers),
                    "gt_heads": int(gt_heads),
                    "best_epoch": int(best_epoch),
                    "best_val": best_metrics,
                },
                ckpt,
            )
        if epoch == 1 or epoch % log_every == 0 or improved:
            metric_text = (
                f"val_mae={val_metrics['mae']:.4g} val_score={val_metrics['score']:.4g}"
                if task == "add"
                else f"val_acc={val_metrics['accuracy']:.4g}"
            )
            print(
                f"[train] task={task} model={model_name} seed={seed} epoch={epoch:04d} "
                f"train_loss={train_loss:.5g} val_loss={val_metrics['loss']:.5g} "
                f"{metric_text} best_epoch={best_epoch}"
            )
        if epoch - best_epoch >= patience:
            print(
                f"[train] early stop task={task} model={model_name} seed={seed} "
                f"epoch={epoch} best_epoch={best_epoch} best_val_loss={best_loss:.5g}"
            )
            break
    print(f"[train] wrote best checkpoint {ckpt}")
    return ckpt


def load_trained_model(
    root: Path,
    task: str,
    model_name: str,
    seed: int,
    *,
    device: torch.device,
) -> tuple[nn.Module, ExperimentSpec, dict[str, Any]]:
    ckpt = checkpoint_path(root, task, model_name, seed)
    payload = torch.load(ckpt, map_location=device)
    spec = ExperimentSpec(**payload["spec"])
    model = build_model(
        model_name,
        task,
        spec,
        hidden_dim=int(payload["hidden_dim"]),
        gt_layers=int(payload["gt_layers"]),
        gt_heads=int(payload["gt_heads"]),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, spec, payload


def run_training(args: argparse.Namespace) -> None:
    spec = spec_from_args(args)
    root = Path(args.output_root)
    data_file = save_dataset(root, spec, overwrite=args.overwrite_data)
    tasks = parse_csv_list(args.tasks, allowed=TASKS)
    models = parse_csv_list(args.models, allowed=MODELS)
    seeds = parse_seed_list(args.seeds)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    for task in tasks:
        for model_name in models:
            for seed in seeds:
                train_one(
                    root=root,
                    data_file=data_file,
                    task=task,
                    model_name=model_name,
                    seed=seed,
                    device=device,
                    hidden_dim=args.hidden_dim,
                    gt_layers=args.gt_layers,
                    gt_heads=args.gt_heads,
                    batch_size=args.batch_size,
                    max_epochs=args.max_epochs,
                    patience=args.patience,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    log_every=args.log_every,
                    overwrite=args.overwrite_checkpoints,
                )


def clean_evaluate(args: argparse.Namespace) -> list[dict[str, Any]]:
    spec = spec_from_args(args)
    root = Path(args.output_root)
    data_file = save_dataset(root, spec, overwrite=False)
    loaded_spec, splits = load_dataset(data_file)
    tasks = parse_csv_list(args.tasks, allowed=TASKS)
    models = parse_csv_list(args.models, allowed=MODELS)
    seeds = parse_seed_list(args.seeds)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    rows: list[dict[str, Any]] = []
    for task in tasks:
        for model_name in models:
            for seed in seeds:
                model, _, payload = load_trained_model(root, task, model_name, seed, device=device)
                metrics = evaluate_model(
                    model,
                    task,
                    splits["test"],
                    loaded_spec,
                    device=device,
                    batch_size=args.eval_batch_size,
                )
                row = {
                    "task": task,
                    "task_label": TASK_LABELS[task],
                    "model": model_name,
                    "model_label": MODEL_LABELS[model_name],
                    "seed": int(seed),
                    "test_loss": metrics["loss"],
                    "test_mae": metrics["mae"],
                    "test_accuracy": metrics["accuracy"],
                    "poster_score": metrics["score"],
                    "best_epoch": payload.get("best_epoch", ""),
                    "best_val_loss": payload.get("best_val", {}).get("loss", ""),
                    "n_test": int(splits["test"]["x"].shape[0]),
                }
                print(
                    f"[eval] task={task} model={model_name} seed={seed} "
                    f"score={metrics['score']:.4g} loss={metrics['loss']:.4g}"
                )
                rows.append(row)
    write_csv(clean_metrics_path(root), rows)
    print(f"[eval] wrote {clean_metrics_path(root)}")
    return rows


def replacement_values(values: torch.Tensor, alphabet_size: int, rng: random.Random) -> torch.Tensor:
    out = []
    alphabet = list(range(alphabet_size))
    for value in values.detach().cpu().tolist():
        choices = [candidate for candidate in alphabet if candidate != int(value)]
        out.append(rng.choice(choices))
    return torch.tensor(out, dtype=torch.long, device=values.device)


def oracle_scalar(task: str, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    if task == "add":
        return (x1 + x2).float()
    return (x1 > x2).float()


@torch.no_grad()
def rq3_for_model(
    *,
    model: nn.Module,
    task: str,
    split: Mapping[str, torch.Tensor],
    spec: ExperimentSpec,
    model_name: str,
    seed: int,
    device: torch.device,
    resamples_per_site: int,
    batch_size: int,
    rq3_seed: int,
) -> list[dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    n = int(split["x"].shape[0])
    rng = random.Random(int(rq3_seed) + int(seed) * 1009 + (0 if task == "add" else 37))
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        p1 = split["p1"][start:end].to(device)
        p2 = split["p2"][start:end].to(device)
        x1 = split["x1"][start:end].to(device)
        x2 = split["x2"][start:end].to(device)
        sample_ids = torch.arange(start, end, dtype=torch.long, device=device)
        if start == 0 or (start // batch_size) % 10 == 0:
            print(
                f"[rq3] task={task} model={model_name} seed={seed} "
                f"examples={start}-{end}/{n}"
            )
        for resample_idx in range(int(resamples_per_site)):
            x1_new = replacement_values(x1, spec.alphabet_size, rng)
            x2_new = replacement_values(x2, spec.alphabet_size, rng)
            x_base = encode_values(p1, p2, x1, x2, spec)
            x_a = encode_values(p1, p2, x1_new, x2, spec)
            x_b = encode_values(p1, p2, x1, x2_new, spec)
            x_ab = encode_values(p1, p2, x1_new, x2_new, spec)
            stacked = torch.cat([x_base, x_a, x_b, x_ab], dim=0).to(device)
            outputs = scalar_output(task, model(stacked))
            bsz = end - start
            f_base, f_a, f_b, f_ab = outputs.split(bsz, dim=0)
            y_base = oracle_scalar(task, x1, x2)
            y_a = oracle_scalar(task, x1_new, x2)
            y_b = oracle_scalar(task, x1, x2_new)
            y_ab = oracle_scalar(task, x1_new, x2_new)
            dy_a = y_a - y_base
            dy_b = y_b - y_base
            dy_ab = y_ab - y_base
            df_a = f_a - f_base
            df_b = f_b - f_base
            df_ab = f_ab - f_base
            oracle_interaction = dy_ab - (dy_a + dy_b)
            model_interaction = df_ab - (df_a + df_b)
            for local_idx in range(bsz):
                rows.append(
                    {
                        "task": task,
                        "task_label": TASK_LABELS[task],
                        "model": model_name,
                        "model_label": MODEL_LABELS[model_name],
                        "seed": int(seed),
                        "sample_id": int(sample_ids[local_idx].item()),
                        "resample_id": int(resample_idx),
                        "receiver_t": 0,
                        "p1": int(p1[local_idx].item()),
                        "p2": int(p2[local_idx].item()),
                        "x1": int(x1[local_idx].item()),
                        "x2": int(x2[local_idx].item()),
                        "x1_new": int(x1_new[local_idx].item()),
                        "x2_new": int(x2_new[local_idx].item()),
                        "y_base": float(y_base[local_idx].item()),
                        "y_a": float(y_a[local_idx].item()),
                        "y_b": float(y_b[local_idx].item()),
                        "y_ab": float(y_ab[local_idx].item()),
                        "f_base": float(f_base[local_idx].item()),
                        "f_a": float(f_a[local_idx].item()),
                        "f_b": float(f_b[local_idx].item()),
                        "f_ab": float(f_ab[local_idx].item()),
                        "delta_a_oracle": float(dy_a[local_idx].item()),
                        "delta_b_oracle": float(dy_b[local_idx].item()),
                        "delta_ab_oracle": float(dy_ab[local_idx].item()),
                        "interaction_oracle": float(oracle_interaction[local_idx].item()),
                        "delta_a_model": float(df_a[local_idx].item()),
                        "delta_b_model": float(df_b[local_idx].item()),
                        "delta_ab_model": float(df_ab[local_idx].item()),
                        "interaction_model": float(model_interaction[local_idx].item()),
                    }
                )
    return rows


def pearson(x: Sequence[float], y: Sequence[float], *, min_std: float = EPS) -> float:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.size < 3 or np.std(x_arr) < min_std or np.std(y_arr) < min_std:
        return float("nan")
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def summarize_rq3_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["task"]), str(row["model"]), int(row["seed"])), []).append(row)
    summary: list[dict[str, Any]] = []
    for (task, model_name, seed), group in sorted(grouped.items()):
        oracle = [float(row["interaction_oracle"]) for row in group]
        model = [float(row["interaction_model"]) for row in group]
        abs_err = [abs(float(row["interaction_model"]) - float(row["interaction_oracle"])) for row in group]
        model_std = float(np.std(model))
        oracle_std = float(np.std(oracle))
        summary.append(
            {
                "task": task,
                "task_label": TASK_LABELS[task],
                "model": model_name,
                "model_label": MODEL_LABELS[model_name],
                "seed": seed,
                "n": len(group),
                "oracle_interaction_mean": float(np.mean(oracle)),
                "oracle_interaction_abs_mean": float(np.mean(np.abs(oracle))),
                "model_interaction_mean": float(np.mean(model)),
                "model_interaction_abs_mean": float(np.mean(np.abs(model))),
                "oracle_interaction_std": oracle_std,
                "model_interaction_std": model_std,
                "interaction_mae": float(np.mean(abs_err)),
                "interaction_pearson": pearson(model, oracle, min_std=INTERACTION_SIGNAL_EPS),
                "interaction_correlation_status": "ok" if model_std >= INTERACTION_SIGNAL_EPS else "no_model_signal",
                "oracle_nonzero_fraction": float(np.mean(np.abs(oracle) > 1.0e-9)),
            }
        )
    return summary


def run_rq3(args: argparse.Namespace) -> list[dict[str, Any]]:
    spec = spec_from_args(args)
    root = Path(args.output_root)
    data_file = save_dataset(root, spec, overwrite=False)
    loaded_spec, splits = load_dataset(data_file)
    tasks = parse_csv_list(args.tasks, allowed=TASKS)
    models = parse_csv_list(args.models, allowed=MODELS)
    seeds = parse_seed_list(args.seeds)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    all_rows: list[dict[str, Any]] = []
    for task in tasks:
        for model_name in models:
            for seed in seeds:
                model, _, _ = load_trained_model(root, task, model_name, seed, device=device)
                all_rows.extend(
                    rq3_for_model(
                        model=model,
                        task=task,
                        split=splits["test"],
                        spec=loaded_spec,
                        model_name=model_name,
                        seed=seed,
                        device=device,
                        resamples_per_site=args.resamples_per_site,
                        batch_size=args.eval_batch_size,
                        rq3_seed=args.rq3_seed,
                    )
                )
    write_csv(rq3_rows_path(root), all_rows)
    summary = summarize_rq3_rows(all_rows)
    write_csv(rq3_summary_path(root), summary)
    print(f"[rq3] wrote rows={rq3_rows_path(root)} summary={rq3_summary_path(root)}")
    return all_rows


def aggregate_mean_sem(rows: Sequence[Mapping[str, Any]], value_key: str) -> dict[tuple[str, str], dict[str, float]]:
    by_key_seed: dict[tuple[str, str], dict[int, list[float]]] = {}
    for row in rows:
        key = (str(row["task"]), str(row["model"]))
        seed = int(row["seed"])
        value = row.get(value_key)
        if value in ("", None):
            continue
        value_f = float(value)
        if math.isnan(value_f):
            continue
        by_key_seed.setdefault(key, {}).setdefault(seed, []).append(value_f)
    out: dict[tuple[str, str], dict[str, float]] = {}
    for key, by_seed in by_key_seed.items():
        seed_means = np.asarray([np.mean(values) for values in by_seed.values()], dtype=np.float64)
        mean = float(seed_means.mean()) if seed_means.size else float("nan")
        sem = float(seed_means.std(ddof=1) / math.sqrt(seed_means.size)) if seed_means.size > 1 else 0.0
        out[key] = {"mean": mean, "sem": sem, "n_seeds": int(seed_means.size)}
    return out


def plot_accuracy_grid(root: Path) -> Path:
    rows = read_csv(clean_metrics_path(root))
    agg = aggregate_mean_sem(rows, "poster_score")
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    present_models = {row["model"] for row in rows}
    present_tasks = {row["task"] for row in rows}
    model_order = [model for model in MODELS if model in present_models]
    task_order = [task for task in TASKS if task in present_tasks]
    width = 0.18
    x = np.arange(len(task_order), dtype=np.float64)
    colors = {
        "mpnn": "#8c8c8c",
        "mpnn_vn": "#4c78a8",
        "gt": "#f58518",
        "nope_gt": "#54a24b",
    }
    for midx, model_name in enumerate(model_order):
        offsets = x + (midx - (len(model_order) - 1) / 2.0) * width
        means = [agg.get((task, model_name), {}).get("mean", np.nan) for task in task_order]
        sems = [agg.get((task, model_name), {}).get("sem", 0.0) for task in task_order]
        ax.bar(
            offsets,
            means,
            width=width,
            yerr=sems,
            capsize=3,
            color=colors[model_name],
            edgecolor="black",
            linewidth=0.6,
            label=MODEL_LABELS[model_name],
        )
    ax.axhline(0.5, color="#666666", linestyle="--", linewidth=0.9, alpha=0.7)
    ax.text(1.44, 0.505, "REL chance", fontsize=8, color="#555555", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABELS[task] for task in task_order])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Performance\nADD: 1 - MAE / label range; REL: accuracy")
    ax.set_title("Minimal Order/Identity Dissociation: Clean Performance")
    ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    fig.tight_layout()
    path = figures_dir(root) / "fig1_accuracy_grid.pdf"
    ensure_dir(path.parent)
    fig.savefig(path)
    plt.close(fig)
    print(f"[plot] wrote {path}")
    return path


def sample_for_plot(values: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if values.size <= max_points:
        return values
    rng = np.random.default_rng(seed)
    idx = rng.choice(values.size, size=max_points, replace=False)
    return values[idx]


def plot_interaction_panel(root: Path) -> Path:
    rows = read_csv(rq3_rows_path(root))
    summary = read_csv(rq3_summary_path(root))
    present_models = {row["model"] for row in rows}
    present_tasks = {row["task"] for row in rows}
    model_order = [model for model in MODELS if model in present_models]
    task_order = [task for task in TASKS if task in present_tasks]
    if not task_order or not model_order:
        raise ValueError("RQ3 rows must contain at least one task and one model")
    fig, axes = plt.subplots(
        2,
        len(task_order),
        figsize=(5.7 * len(task_order), 7.0),
        squeeze=False,
        gridspec_kw={"height_ratios": [2.5, 1.0]},
    )
    row_by_task_model: dict[tuple[str, str], list[dict[str, str]]] = {}
    oracle_by_task: dict[str, list[float]] = {}
    for row in rows:
        task = row["task"]
        model_name = row["model"]
        row_by_task_model.setdefault((task, model_name), []).append(row)
        oracle_by_task.setdefault(task, []).append(float(row["interaction_oracle"]))
    corr_by_task_model = {
        (row["task"], row["model"]): float(row["interaction_pearson"])
        for row in summary
        if row.get("interaction_pearson") not in ("", "nan")
    }
    colors = ["#bdbdbd", "#4c78a8", "#f58518", "#54a24b"]
    for col, task in enumerate(task_order):
        ax = axes[0, col]
        data: list[np.ndarray] = []
        labels: list[str] = []
        oracle_vals = np.asarray(oracle_by_task.get(task, []), dtype=np.float64)
        if oracle_vals.size:
            data.append(sample_for_plot(oracle_vals, 8000, 11 + col))
            labels.append("Oracle")
        for model_name in model_order:
            vals = np.asarray(
                [float(row["interaction_model"]) for row in row_by_task_model.get((task, model_name), [])],
                dtype=np.float64,
            )
            if vals.size:
                data.append(sample_for_plot(vals, 8000, 101 + col))
                labels.append(MODEL_LABELS[model_name])
        parts = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False)
        for body_idx, body in enumerate(parts["bodies"]):
            body.set_alpha(0.78)
            body.set_facecolor("#111111" if body_idx == 0 else colors[body_idx - 1])
            body.set_edgecolor("black")
            body.set_linewidth(0.5)
        for idx, vals in enumerate(data, start=1):
            if vals.size == 0:
                continue
            q1, med, q3 = np.percentile(vals, [25, 50, 75])
            ax.plot([idx - 0.18, idx + 0.18], [med, med], color="black", linewidth=1.3)
            ax.plot([idx, idx], [q1, q3], color="black", linewidth=2.2)
        ax.axhline(0.0, color="#555555", linestyle="--", linewidth=0.9)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_title(f"{TASK_LABELS[task]} RQ3 interaction")
        ax.set_ylabel("interaction = delta_AB - (delta_A + delta_B)")
        ax.grid(axis="y", color="#e0e0e0", linewidth=0.6)
    for col, task in enumerate(task_order):
        ax = axes[1, col]
        x = np.arange(len(model_order), dtype=np.float64)
        vals = [corr_by_task_model.get((task, model_name), np.nan) for model_name in model_order]
        ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.6)
        for xpos, val in zip(x, vals):
            if math.isnan(float(val)):
                ax.text(
                    xpos,
                    0.02,
                    "no\nsignal",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#444444",
                )
        ax.axhline(0.0, color="#555555", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[name] for name in model_order], rotation=20, ha="right")
        ax.set_ylim(-1.05, 1.05)
        ax.set_ylabel("corr(model, oracle)")
        ax.set_title(f"{TASK_LABELS[task]} interaction-oracle correlation")
        ax.grid(axis="y", color="#e0e0e0", linewidth=0.6)
    fig.suptitle("Minimal Order/Identity Dissociation: RQ3 Output Interaction", y=0.995)
    fig.tight_layout()
    path = figures_dir(root) / "fig2_interaction_panel.pdf"
    ensure_dir(path.parent)
    fig.savefig(path)
    plt.close(fig)
    print(f"[plot] wrote {path}")
    return path


def plot_schematic(root: Path, spec: ExperimentSpec) -> Path:
    fig, ax = plt.subplots(figsize=(10.5, 2.8))
    y = np.zeros(spec.n_nodes)
    x = np.arange(spec.n_nodes)
    ax.plot(x, y, color="#888888", linewidth=1.4, zorder=1)
    ax.scatter(x, y, s=45, color="#d9d9d9", edgecolor="#555555", linewidth=0.5, zorder=2)
    ax.scatter([0], [0], s=130, color="#111111", edgecolor="black", zorder=4)
    ax.text(0, 0.18, "t = 0\nreadout", ha="center", va="bottom", fontsize=10, weight="bold")
    far_start = spec.position_low
    ax.axvspan(0.5, spec.receptive_radius + 0.5, color="#f2f2f2", alpha=1.0, zorder=0)
    ax.text(spec.receptive_radius / 2.0, -0.23, "MPNN radius r", ha="center", va="top", fontsize=9)
    ax.axvspan(far_start - 0.5, spec.position_high + 0.5, color="#fff2cc", alpha=0.7, zorder=0)
    example_p1 = far_start + 4
    example_p2 = min(spec.n_nodes - 2, far_start + 14)
    ax.scatter([example_p1, example_p2], [0, 0], s=150, color="#f58518", edgecolor="black", zorder=5)
    ax.text(example_p1, 0.2, "p1\nvalue x_p1", ha="center", va="bottom", fontsize=10)
    ax.text(example_p2, 0.2, "p2\nvalue x_p2", ha="center", va="bottom", fontsize=10)
    ax.annotate(
        "sample p1 < p2 uniformly from [r+2, N-1]\nno role flags; only values are marked",
        xy=((example_p1 + example_p2) / 2, 0),
        xytext=((example_p1 + example_p2) / 2, -0.55),
        ha="center",
        va="top",
        arrowprops={"arrowstyle": "-|>", "color": "#555555", "lw": 0.8},
        fontsize=10,
    )
    ax.text(
        spec.n_nodes - 1,
        0.18,
        "ADD: x_p1 + x_p2\nREL: 1[x_p1 > x_p2]",
        ha="right",
        va="bottom",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#bbbbbb"},
    )
    ax.set_xlim(-0.8, spec.n_nodes - 0.2)
    ax.set_ylim(-0.75, 0.55)
    ax.set_yticks([])
    ax.set_xlabel("path position / distance from readout endpoint")
    ax.set_title("Minimal Order/Identity Dissociation Setup")
    for spine in ("left", "right", "top"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    path = figures_dir(root) / "fig3_schematic.pdf"
    ensure_dir(path.parent)
    fig.savefig(path)
    plt.close(fig)
    print(f"[plot] wrote {path}")
    return path


def plot_all(args: argparse.Namespace) -> None:
    spec = spec_from_args(args)
    root = Path(args.output_root)
    plot_accuracy_grid(root)
    plot_interaction_panel(root)
    plot_schematic(root, spec)


def spec_from_args(args: argparse.Namespace) -> ExperimentSpec:
    return ExperimentSpec(
        n_nodes=int(args.n_nodes),
        receptive_radius=int(args.receptive_radius),
        alphabet_size=int(args.alphabet_size),
        train_size=int(args.train_size),
        val_size=int(args.val_size),
        test_size=int(args.test_size),
        data_seed=int(args.data_seed),
    )


def build_data_command(args: argparse.Namespace) -> None:
    spec = spec_from_args(args)
    save_dataset(Path(args.output_root), spec, overwrite=args.overwrite_data)
    write_json(
        config_record_path(Path(args.output_root)),
        {
            "spec": asdict(spec),
            "models": parse_csv_list(args.models, allowed=MODELS),
            "tasks": parse_csv_list(args.tasks, allowed=TASKS),
            "seeds": parse_seed_list(args.seeds),
            "hidden_dim": int(args.hidden_dim),
            "gt_layers": int(args.gt_layers),
            "gt_heads": int(args.gt_heads),
            "resamples_per_site": int(args.resamples_per_site),
        },
    )


def run_all(args: argparse.Namespace) -> None:
    build_data_command(args)
    run_training(args)
    clean_evaluate(args)
    run_rq3(args)
    plot_all(args)
    print(f"[done] minimal order/identity experiment complete: {Path(args.output_root)}")


def print_hpc_commands(args: argparse.Namespace) -> None:
    root = Path(args.output_root)
    common = (
        "source /usr/local/Cluster-Apps/miniconda3/4.5.1/etc/profile.d/conda.sh; "
        "conda activate graphbench-algoreas; "
        "export PYTHONPATH=$PWD/src:$PYTHONPATH; "
        "export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8}; "
    )
    module = "python -u -m graph_specialisation_metrics.minimal_order_dissociation_experiment"
    flags = (
        f"--output-root {root} --tasks {args.tasks} --models {args.models} --seeds {args.seeds} "
        f"--n-nodes {args.n_nodes} --receptive-radius {args.receptive_radius} "
        f"--train-size {args.train_size} --val-size {args.val_size} --test-size {args.test_size} "
        f"--hidden-dim {args.hidden_dim} --gt-layers {args.gt_layers} --gt-heads {args.gt_heads} "
        f"--max-epochs {args.max_epochs} --patience {args.patience} "
        f"--batch-size {args.batch_size} --eval-batch-size {args.eval_batch_size} "
        f"--resamples-per-site {args.resamples_per_site} --log-every {args.log_every}"
    )
    wrap = common + module + " run-all " + flags + " --device cuda"
    print("mkdir -p logs")
    print(
        "sbatch -A mlmi-jgg45-sl2-gpu -p ampere --qos=gpu1 "
        "--gres=gpu:1 --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=80G --time=01:00:00 "
        "-J min-order-full -o logs/min-order-full-%j.out -e logs/min-order-full-%j.err "
        f"--wrap {json.dumps(wrap)}"
    )


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-root", type=Path, default=default_output_root())
    parser.add_argument("--tasks", type=str, default="add,rel")
    parser.add_argument("--models", type=str, default="mpnn,mpnn_vn,gt,nope_gt")
    parser.add_argument("--seeds", type=str, default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--n-nodes", type=int, default=32)
    parser.add_argument("--receptive-radius", type=int, default=4)
    parser.add_argument("--alphabet-size", type=int, default=4)
    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--data-seed", type=int, default=7001)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--gt-layers", type=int, default=3)
    parser.add_argument("--gt-heads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--max-epochs", type=int, default=800)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--resamples-per-site", type=int, default=4)
    parser.add_argument("--rq3-seed", type=int, default=8101)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--overwrite-data", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--overwrite-checkpoints", action=argparse.BooleanOptionalAction, default=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, help_text, fn in [
        ("build-data", "Generate and cache the path-order dataset.", build_data_command),
        ("train", "Train selected models/tasks/seeds.", run_training),
        ("evaluate", "Evaluate clean held-out performance.", clean_evaluate),
        ("run-rq3", "Run the output-level double-intervention additivity test.", run_rq3),
        ("plot", "Generate all poster figures from cached metrics.", plot_all),
        ("run-all", "Run data, training, clean eval, RQ3, and figures.", run_all),
        ("print-hpc-commands", "Print a full-chain sbatch command.", print_hpc_commands),
    ]:
        sub = subparsers.add_parser(name, help=help_text)
        add_common_args(sub)
        sub.set_defaults(func=fn)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
