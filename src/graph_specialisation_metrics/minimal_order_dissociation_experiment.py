"""Minimal experiment v2: positional identity in long-range coupling.

This runner implements ``minimal_experiment_v2.md`` as a standalone synthetic
experiment.  It trains a matched ladder of local, symmetric-global, content-only
dense, routing-structured, and full structural-value models on three continuous
path-graph tasks, then runs the RQ3 double-resample interaction instrument and a
GT-full channel-clamp attribution check.
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


TASKS = ("add", "prod", "gap")
TASK_LABELS = {"add": "ADD", "prod": "PROD", "gap": "GAP"}
DEFAULT_MODELS = ("mpnn", "mpnn_vn", "nope_gt", "gt_route", "gt_full")
ALL_MODELS = DEFAULT_MODELS + ("gt_pe_anchor",)
MODEL_LABELS = {
    "mpnn": "MPNN",
    "mpnn_vn": "MPNN+VN",
    "nope_gt": "NoPE-GT",
    "gt_route": "GT-route",
    "gt_full": "GT-full",
    "gt_pe_anchor": "PE-value anchor",
}
DEFAULT_SEEDS = (1001, 1002, 1003)
EPS = 1.0e-12
INTERACTION_SIGNAL_EPS = 1.0e-5


@dataclass(frozen=True)
class ExperimentSpec:
    n_nodes: int = 32
    receptive_radius: int = 4
    train_size: int = 5000
    val_size: int = 1000
    test_size: int = 1000
    data_seed: int = 7001

    @property
    def value_feature_dim(self) -> int:
        # [is_value_node, continuous_value].  This marks non-null nodes but does
        # not reveal whether a value is p1 or p2.
        return 2

    @property
    def position_low(self) -> int:
        return self.receptive_radius + 2

    @property
    def position_high(self) -> int:
        return self.n_nodes - 1

    @property
    def label_range(self) -> dict[str, float]:
        return {"add": 2.0, "prod": 1.0, "gap": 1.0}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_output_root() -> Path:
    return Path.cwd() / "artifacts" / "minimal_order_dissociation_v2"


def data_path(root: Path, spec: ExperimentSpec) -> Path:
    return root / "data" / f"path_identity_v2_N{spec.n_nodes}_r{spec.receptive_radius}_seed{spec.data_seed}.pt"


def checkpoint_path(root: Path, task: str, model_name: str, seed: int) -> Path:
    return root / "checkpoints" / task / model_name / f"seed_{int(seed)}" / "best.pt"


def metrics_dir(root: Path) -> Path:
    return root / "metrics"


def figures_dir(root: Path) -> Path:
    return root / "figures"


def clean_metrics_path(root: Path) -> Path:
    return metrics_dir(root) / "clean_performance.csv"


def rq3_rows_path(root: Path) -> Path:
    return metrics_dir(root) / "rq3_interactions.csv"


def rq3_summary_path(root: Path) -> Path:
    return metrics_dir(root) / "rq3_summary.csv"


def channel_rows_path(root: Path) -> Path:
    return metrics_dir(root) / "channel_attribution_rows.csv"


def channel_summary_path(root: Path) -> Path:
    return metrics_dir(root) / "channel_attribution_summary.csv"


def config_record_path(root: Path) -> Path:
    return metrics_dir(root) / "configuration.json"


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    ensure_dir(path.parent)
    keys: list[str] = []
    if fieldnames is None:
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


def encode_values(
    p1: torch.Tensor,
    p2: torch.Tensor,
    x1: torch.Tensor,
    x2: torch.Tensor,
    spec: ExperimentSpec,
) -> torch.Tensor:
    batch_size = int(p1.numel())
    x = torch.zeros(batch_size, spec.n_nodes, spec.value_feature_dim, dtype=torch.float32, device=p1.device)
    rows = torch.arange(batch_size, device=p1.device)
    x[rows, p1.long(), 0] = 1.0
    x[rows, p2.long(), 0] = 1.0
    x[rows, p1.long(), 1] = x1.float()
    x[rows, p2.long(), 1] = x2.float()
    return x


def oracle_scalar(task: str, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    if task == "add":
        return x1.float() + x2.float()
    if task == "prod":
        return x1.float() * x2.float()
    if task == "gap":
        return torch.clamp(x1.float() - x2.float(), min=0.0)
    raise ValueError(f"unknown task {task!r}")


def make_split(spec: ExperimentSpec, count: int, seed: int) -> dict[str, torch.Tensor]:
    rng = random.Random(int(seed))
    torch_gen = torch.Generator().manual_seed(int(seed) + 17)
    candidates = list(range(spec.position_low, spec.position_high + 1))
    p1_rows: list[int] = []
    p2_rows: list[int] = []
    for _ in range(int(count)):
        p1, p2 = sorted(rng.sample(candidates, 2))
        p1_rows.append(p1)
        p2_rows.append(p2)
    p1_t = torch.tensor(p1_rows, dtype=torch.long)
    p2_t = torch.tensor(p2_rows, dtype=torch.long)
    x1_t = torch.rand(int(count), generator=torch_gen, dtype=torch.float32)
    x2_t = torch.rand(int(count), generator=torch_gen, dtype=torch.float32)
    return {
        "x": encode_values(p1_t, p2_t, x1_t, x2_t, spec),
        "p1": p1_t,
        "p2": p2_t,
        "x1": x1_t,
        "x2": x2_t,
        "y_add": oracle_scalar("add", x1_t, x2_t),
        "y_prod": oracle_scalar("prod", x1_t, x2_t),
        "y_gap": oracle_scalar("gap", x1_t, x2_t),
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
        if not bool(torch.all((x1 >= 0.0) & (x1 <= 1.0) & (x2 >= 0.0) & (x2 <= 1.0))):
            raise ValueError(f"{split_name}: values must lie in [0, 1]")
        if not torch.allclose(split["y_add"], oracle_scalar("add", x1, x2)):
            raise ValueError(f"{split_name}: ADD labels are inconsistent")
        if not torch.allclose(split["y_prod"], oracle_scalar("prod", x1, x2)):
            raise ValueError(f"{split_name}: PROD labels are inconsistent")
        if not torch.allclose(split["y_gap"], oracle_scalar("gap", x1, x2)):
            raise ValueError(f"{split_name}: GAP labels are inconsistent")
        marker_count = split["x"][:, :, 0].sum(dim=-1)
        if not torch.allclose(marker_count, torch.full_like(marker_count, 2.0)):
            raise ValueError(f"{split_name}: exactly two non-null nodes are required")


def build_dataset(spec: ExperimentSpec) -> dict[str, Any]:
    train = make_split(spec, spec.train_size, spec.data_seed + 101)
    val = make_split(spec, spec.val_size, spec.data_seed + 202)
    test = make_split(spec, spec.test_size, spec.data_seed + 303)
    payload = {"spec": asdict(spec), "splits": {"train": train, "val": val, "test": test}}
    validate_dataset(payload, spec)
    return payload


def save_dataset(root: Path, spec: ExperimentSpec, *, overwrite: bool = False) -> Path:
    path = data_path(root, spec)
    if path.exists() and not overwrite:
        print(f"[data] using existing cache {path}")
        return path
    ensure_dir(path.parent)
    torch.save(build_dataset(spec), path)
    print(
        f"[data] wrote {path} train={spec.train_size} val={spec.val_size} "
        f"test={spec.test_size} N={spec.n_nodes} r={spec.receptive_radius}"
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
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LocalMPNN(nn.Module):
    """Depth-r local MPNN.  The receiver cannot access p1/p2 by construction."""

    def __init__(self, *, n_nodes: int, in_dim: int, hidden_dim: int, depth: int):
        super().__init__()
        self.encoder = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([ResidualMLP(hidden_dim * 2, hidden_dim * 2, hidden_dim) for _ in range(depth)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(depth)])
        self.readout = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        adj = path_adjacency(n_nodes)
        deg = adj.sum(dim=-1).clamp_min(1.0)
        self.register_buffer("adj_norm", adj / deg[:, None], persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        adj = self.adj_norm.to(device=x.device, dtype=h.dtype)
        for layer, norm in zip(self.layers, self.norms):
            neigh = torch.einsum("ij,bjh->bih", adj, h)
            h = norm(h + layer(torch.cat([h, neigh], dim=-1)))
        return self.readout(h[:, 0, :]).reshape(-1)


class SymmetricGlobalReadout(nn.Module):
    """Symmetric full-graph summary used for the MPNN+VN control."""

    def __init__(self, *, in_dim: int, hidden_dim: int):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.readout = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.phi(x).sum(dim=1)
        return self.readout(pooled).reshape(-1)


class ContentOnlyDenseReadout(nn.Module):
    """NoPE dense content readout: full reach, no positional or distance input."""

    def __init__(self, *, in_dim: int, hidden_dim: int, num_heads: int):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.num_heads = int(num_heads)
        self.head_dim = hidden_dim // num_heads
        self.content = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.query = nn.Parameter(torch.randn(num_heads, self.head_dim) * 0.02)
        self.readout = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.content(x)
        bsz, n_nodes, hidden = h.shape
        k = self.key(h).view(bsz, n_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(h).view(bsz, n_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.einsum("hd,bhnd->bhn", self.query.to(x.device), k) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        context = torch.einsum("bhn,bhnd->bhd", attn, v).reshape(bsz, hidden)
        return self.readout(context).reshape(-1)


class StructuralTransportReadout(nn.Module):
    """GT-route/GT-full receiver readout with named routing and value channels.

    GT-route uses learned relative-distance routing bias but content-only values.
    GT-full keeps the same routing and adds a structural value channel
    ``Vs = x_value * E_position``.  Channel attribution can clamp ``Vs`` to the
    clean value and clamp routing bias to its per-head mean.
    """

    def __init__(
        self,
        *,
        n_nodes: int,
        in_dim: int,
        hidden_dim: int,
        num_heads: int,
        use_structural_value: bool,
        use_routing_bias: bool = True,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.n_nodes = int(n_nodes)
        self.num_heads = int(num_heads)
        self.head_dim = hidden_dim // num_heads
        self.hidden_dim = int(hidden_dim)
        self.use_structural_value = bool(use_structural_value)
        self.use_routing_bias = bool(use_routing_bias)
        self.content = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value_content = nn.Linear(hidden_dim, hidden_dim)
        self.struct_value = nn.Embedding(n_nodes, hidden_dim)
        self.distance_bias = nn.Embedding(n_nodes, num_heads)
        self.query = nn.Parameter(torch.randn(num_heads, self.head_dim) * 0.02)
        self.readout = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.register_buffer("positions", torch.arange(n_nodes, dtype=torch.long), persistent=False)
        self.register_buffer("receiver_dist", torch.arange(n_nodes, dtype=torch.long), persistent=False)

    def routing_bias_channel(self, *, device: torch.device, dtype: torch.dtype, clamp_mean: bool = False) -> torch.Tensor:
        if not self.use_routing_bias:
            return torch.zeros(self.num_heads, self.n_nodes, device=device, dtype=dtype)
        bias = self.distance_bias(self.receiver_dist.to(device)).transpose(0, 1).to(dtype=dtype)
        if clamp_mean:
            bias = bias.mean(dim=-1, keepdim=True).expand_as(bias)
        return bias

    def value_struct_channel(self, x: torch.Tensor) -> torch.Tensor:
        pos = self.positions.to(x.device)
        struct = self.struct_value(pos)[None, :, :].to(dtype=x.dtype)
        value = x[:, :, 1:2]
        marker = x[:, :, 0:1]
        return marker * value * struct

    def forward(
        self,
        x: torch.Tensor,
        *,
        clamp_vs_to: torch.Tensor | None = None,
        clamp_routing_bias: bool = False,
    ) -> torch.Tensor:
        h = self.content(x)
        bsz, n_nodes, hidden = h.shape
        k = self.key(h).view(bsz, n_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        v_content = self.value_content(h)
        if self.use_structural_value:
            v_struct = self.value_struct_channel(x) if clamp_vs_to is None else clamp_vs_to.to(device=x.device, dtype=x.dtype)
        else:
            v_struct = torch.zeros_like(v_content)
        v = (v_content + v_struct).view(bsz, n_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.einsum("hd,bhnd->bhn", self.query.to(x.device), k) / math.sqrt(self.head_dim)
        scores = scores + self.routing_bias_channel(device=x.device, dtype=scores.dtype, clamp_mean=clamp_routing_bias)[None, :, :]
        attn = torch.softmax(scores, dim=-1)
        context = torch.einsum("bhn,bhnd->bhd", attn, v).reshape(bsz, hidden)
        return self.readout(context).reshape(-1)


def build_model(model_name: str, spec: ExperimentSpec, *, hidden_dim: int, gt_heads: int) -> nn.Module:
    if model_name == "mpnn":
        return LocalMPNN(
            n_nodes=spec.n_nodes,
            in_dim=spec.value_feature_dim,
            hidden_dim=hidden_dim,
            depth=spec.receptive_radius,
        )
    if model_name == "mpnn_vn":
        return SymmetricGlobalReadout(in_dim=spec.value_feature_dim, hidden_dim=hidden_dim)
    if model_name == "nope_gt":
        return ContentOnlyDenseReadout(in_dim=spec.value_feature_dim, hidden_dim=hidden_dim, num_heads=gt_heads)
    if model_name == "gt_route":
        return StructuralTransportReadout(
            n_nodes=spec.n_nodes,
            in_dim=spec.value_feature_dim,
            hidden_dim=hidden_dim,
            num_heads=gt_heads,
            use_structural_value=False,
            use_routing_bias=True,
        )
    if model_name in {"gt_full", "gt_pe_anchor"}:
        return StructuralTransportReadout(
            n_nodes=spec.n_nodes,
            in_dim=spec.value_feature_dim,
            hidden_dim=hidden_dim,
            num_heads=gt_heads,
            use_structural_value=True,
            use_routing_bias=True,
        )
    raise ValueError(f"unknown model {model_name!r}")


def task_loss(task: str, output: torch.Tensor, split: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return F.mse_loss(output.reshape(-1), split[f"y_{task}"].to(output.device))


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
    errors: list[torch.Tensor] = []
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    n = int(split["x"].shape[0])
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = {key: value[start:end].to(device) for key, value in split.items()}
        output = model(batch["x"]).reshape(-1)
        target = batch[f"y_{task}"].reshape(-1)
        loss = F.mse_loss(output, target)
        losses.append(float(loss.item()) * (end - start))
        errors.append((output - target).abs().detach().cpu())
        preds.append(output.detach().cpu())
        targets.append(target.detach().cpu())
    pred = torch.cat(preds) if preds else torch.empty(0)
    target = torch.cat(targets) if targets else torch.empty(0)
    abs_err = torch.cat(errors) if errors else torch.empty(0)
    mae = float(abs_err.mean().item()) if abs_err.numel() else float("nan")
    mse = float(((pred - target) ** 2).mean().item()) if pred.numel() else float("nan")
    var = float(torch.var(target, unbiased=False).item()) if target.numel() else float("nan")
    r2 = 1.0 - mse / max(var, EPS)
    score = 1.0 - mae / max(spec.label_range[task], EPS)
    return {"loss": sum(losses) / max(n, 1), "mae": mae, "mse": mse, "r2": r2, "score": score}


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
    model = build_model(model_name, spec, hidden_dim=hidden_dim, gt_heads=gt_heads).to(device)
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
            loss = task_loss(task, model(batch["x"]), batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total += float(loss.item()) * int(batch["x"].shape[0])
            count += int(batch["x"].shape[0])
        val_metrics = evaluate_model(model, task, splits["val"], spec, device=device, batch_size=batch_size)
        train_loss = total / max(count, 1)
        improved = val_metrics["loss"] < best_loss - 1.0e-8
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
                    "gt_heads": int(gt_heads),
                    "best_epoch": int(best_epoch),
                    "best_val": best_metrics,
                },
                ckpt,
            )
        if epoch == 1 or epoch % log_every == 0 or improved:
            print(
                f"[train] task={task} model={model_name} seed={seed} epoch={epoch:04d} "
                f"train_loss={train_loss:.6g} val_loss={val_metrics['loss']:.6g} "
                f"val_mae={val_metrics['mae']:.5g} val_r2={val_metrics['r2']:.5g} "
                f"val_score={val_metrics['score']:.5g} best_epoch={best_epoch}"
            )
        if epoch - best_epoch >= patience:
            print(
                f"[train] early stop task={task} model={model_name} seed={seed} "
                f"epoch={epoch} best_epoch={best_epoch} best_val_loss={best_loss:.6g}"
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
    payload = torch.load(checkpoint_path(root, task, model_name, seed), map_location=device)
    spec = ExperimentSpec(**payload["spec"])
    model = build_model(model_name, spec, hidden_dim=int(payload["hidden_dim"]), gt_heads=int(payload["gt_heads"])).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, spec, payload


def replacement_values(values: torch.Tensor, rng: torch.Generator) -> torch.Tensor:
    return torch.rand(values.shape, generator=rng, dtype=torch.float32, device=values.device)


def pearson(x: Sequence[float], y: Sequence[float], *, min_std: float = EPS) -> float:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.size < 3 or np.std(x_arr) < min_std or np.std(y_arr) < min_std:
        return float("nan")
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


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
    rng = torch.Generator(device=device).manual_seed(int(rq3_seed) + int(seed) * 1009 + 37 * TASKS.index(task))
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        p1 = split["p1"][start:end].to(device)
        p2 = split["p2"][start:end].to(device)
        x1 = split["x1"][start:end].to(device)
        x2 = split["x2"][start:end].to(device)
        sample_ids = torch.arange(start, end, dtype=torch.long, device=device)
        if start == 0 or (start // batch_size) % 10 == 0:
            print(f"[rq3] task={task} model={model_name} seed={seed} examples={start}-{end}/{n}")
        for resample_idx in range(int(resamples_per_site)):
            x1_new = replacement_values(x1, rng)
            x2_new = replacement_values(x2, rng)
            x_base = encode_values(p1, p2, x1, x2, spec)
            x_a = encode_values(p1, p2, x1_new, x2, spec)
            x_b = encode_values(p1, p2, x1, x2_new, spec)
            x_ab = encode_values(p1, p2, x1_new, x2_new, spec)
            stacked = torch.cat([x_base, x_a, x_b, x_ab], dim=0).to(device)
            f_base, f_a, f_b, f_ab = model(stacked).split(end - start, dim=0)
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
            for local_idx in range(end - start):
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
                        "x1": float(x1[local_idx].item()),
                        "x2": float(x2[local_idx].item()),
                        "x1_new": float(x1_new[local_idx].item()),
                        "x2_new": float(x2_new[local_idx].item()),
                        "on_manifold": True,
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
        corr = float("nan") if task == "add" else pearson(model, oracle, min_std=INTERACTION_SIGNAL_EPS)
        status = "not_applicable_zero_oracle" if task == "add" else ("ok" if model_std >= INTERACTION_SIGNAL_EPS else "no_model_signal")
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
                "interaction_pearson": corr,
                "interaction_correlation_status": status,
                "oracle_nonzero_fraction": float(np.mean(np.abs(oracle) > 1.0e-9)),
            }
        )
    return summary


@torch.no_grad()
def channel_attribution_for_model(
    *,
    model: nn.Module,
    task: str,
    split: Mapping[str, torch.Tensor],
    spec: ExperimentSpec,
    seed: int,
    device: torch.device,
    resamples_per_site: int,
    batch_size: int,
    rq3_seed: int,
    min_oracle_effect: float,
) -> list[dict[str, Any]]:
    if not isinstance(model, StructuralTransportReadout) or not model.use_structural_value:
        raise TypeError("channel attribution requires GT-full StructuralTransportReadout")
    model.eval()
    rows: list[dict[str, Any]] = []
    rng = torch.Generator(device=device).manual_seed(int(rq3_seed) + int(seed) * 2003 + 97 * TASKS.index(task))
    n = int(split["x"].shape[0])
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        p1 = split["p1"][start:end].to(device)
        p2 = split["p2"][start:end].to(device)
        x1 = split["x1"][start:end].to(device)
        x2 = split["x2"][start:end].to(device)
        sample_ids = torch.arange(start, end, dtype=torch.long, device=device)
        if start == 0 or (start // batch_size) % 10 == 0:
            print(f"[channel] task={task} seed={seed} examples={start}-{end}/{n}")
        for resample_idx in range(int(resamples_per_site)):
            x1_new = replacement_values(x1, rng)
            x2_new = replacement_values(x2, rng)
            x_base = encode_values(p1, p2, x1, x2, spec).to(device)
            x_ab = encode_values(p1, p2, x1_new, x2_new, spec).to(device)
            y_base = oracle_scalar(task, x1, x2)
            y_ab = oracle_scalar(task, x1_new, x2_new)
            oracle_delta = y_ab - y_base
            f_base = model(x_base)
            f_ab = model(x_ab)
            vs_clean = model.value_struct_channel(x_base)
            f_vs = model(x_ab, clamp_vs_to=vs_clean)
            f_b = model(x_ab, clamp_routing_bias=True)
            full_delta = f_ab - f_base
            vs_delta = f_vs - f_base
            b_delta = f_b - f_base
            for local_idx in range(end - start):
                certified = abs(float(oracle_delta[local_idx].item())) >= float(min_oracle_effect)
                for clamp_name, clamp_delta in [("Vs-clean-clamp", vs_delta), ("B-mean-clamp", b_delta)]:
                    full = float(full_delta[local_idx].item())
                    clamped = float(clamp_delta[local_idx].item())
                    rows.append(
                        {
                            "task": task,
                            "task_label": TASK_LABELS[task],
                            "model": "gt_full",
                            "model_label": MODEL_LABELS["gt_full"],
                            "seed": int(seed),
                            "sample_id": int(sample_ids[local_idx].item()),
                            "resample_id": int(resample_idx),
                            "clamp": clamp_name,
                            "p1": int(p1[local_idx].item()),
                            "p2": int(p2[local_idx].item()),
                            "oracle_delta_ab": float(oracle_delta[local_idx].item()),
                            "full_delta_ab": full,
                            "clamped_delta_ab": clamped,
                            "abs_survival": abs(clamped) / max(abs(full), EPS),
                            "signed_survival": clamped / full if abs(full) > EPS else float("nan"),
                            "certified_oracle_effect": certified,
                        }
                    )
    return rows


def summarize_channel_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        if str(row.get("certified_oracle_effect", "")).lower() not in {"true", "1"} and row.get("certified_oracle_effect") is not True:
            continue
        grouped.setdefault((str(row["task"]), str(row["clamp"]), int(row["seed"])), []).append(row)
    summary: list[dict[str, Any]] = []
    for (task, clamp, seed), group in sorted(grouped.items()):
        survival = np.asarray([float(row["abs_survival"]) for row in group], dtype=np.float64)
        signed = np.asarray([float(row["signed_survival"]) for row in group if str(row.get("signed_survival")) != "nan"], dtype=np.float64)
        summary.append(
            {
                "task": task,
                "task_label": TASK_LABELS[task],
                "model": "gt_full",
                "model_label": MODEL_LABELS["gt_full"],
                "seed": seed,
                "clamp": clamp,
                "n_certified": int(len(group)),
                "abs_survival_mean": float(np.mean(survival)) if survival.size else float("nan"),
                "abs_survival_median": float(np.median(survival)) if survival.size else float("nan"),
                "signed_survival_mean": float(np.mean(signed)) if signed.size else float("nan"),
            }
        )
    return summary


def aggregate_mean_sem(rows: Sequence[Mapping[str, Any]], value_key: str, key_fields: Sequence[str]) -> dict[tuple[str, ...], dict[str, float]]:
    by_key_seed: dict[tuple[str, ...], dict[int, list[float]]] = {}
    for row in rows:
        value = row.get(value_key)
        if value in ("", None, "nan"):
            continue
        value_f = float(value)
        if math.isnan(value_f):
            continue
        key = tuple(str(row[field]) for field in key_fields)
        seed = int(row["seed"])
        by_key_seed.setdefault(key, {}).setdefault(seed, []).append(value_f)
    out: dict[tuple[str, ...], dict[str, float]] = {}
    for key, by_seed in by_key_seed.items():
        seed_means = np.asarray([np.mean(values) for values in by_seed.values()], dtype=np.float64)
        out[key] = {
            "mean": float(seed_means.mean()) if seed_means.size else float("nan"),
            "sem": float(seed_means.std(ddof=1) / math.sqrt(seed_means.size)) if seed_means.size > 1 else 0.0,
            "n_seeds": int(seed_means.size),
        }
    return out


def run_training(args: argparse.Namespace) -> None:
    spec = spec_from_args(args)
    root = Path(args.output_root)
    data_file = save_dataset(root, spec, overwrite=args.overwrite_data)
    tasks = parse_csv_list(args.tasks, allowed=TASKS)
    models = parse_csv_list(args.models, allowed=ALL_MODELS)
    seeds = parse_seed_list(args.seeds)
    device = resolve_device(args.device)
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
    models = parse_csv_list(args.models, allowed=ALL_MODELS)
    seeds = parse_seed_list(args.seeds)
    device = resolve_device(args.device)
    rows: list[dict[str, Any]] = []
    for task in tasks:
        for model_name in models:
            for seed in seeds:
                model, _, payload = load_trained_model(root, task, model_name, seed, device=device)
                metrics = evaluate_model(model, task, splits["test"], loaded_spec, device=device, batch_size=args.eval_batch_size)
                row = {
                    "task": task,
                    "task_label": TASK_LABELS[task],
                    "model": model_name,
                    "model_label": MODEL_LABELS[model_name],
                    "seed": int(seed),
                    "test_loss": metrics["loss"],
                    "test_mae": metrics["mae"],
                    "test_mse": metrics["mse"],
                    "test_r2": metrics["r2"],
                    "poster_score": metrics["score"],
                    "best_epoch": payload.get("best_epoch", ""),
                    "best_val_loss": payload.get("best_val", {}).get("loss", ""),
                    "n_test": int(splits["test"]["x"].shape[0]),
                }
                print(
                    f"[eval] task={task} model={model_name} seed={seed} "
                    f"mae={metrics['mae']:.5g} r2={metrics['r2']:.5g} score={metrics['score']:.5g}"
                )
                rows.append(row)
    write_csv(clean_metrics_path(root), rows)
    print(f"[eval] wrote {clean_metrics_path(root)}")
    return rows


def run_rq3(args: argparse.Namespace) -> list[dict[str, Any]]:
    spec = spec_from_args(args)
    root = Path(args.output_root)
    data_file = save_dataset(root, spec, overwrite=False)
    loaded_spec, splits = load_dataset(data_file)
    tasks = parse_csv_list(args.tasks, allowed=TASKS)
    models = parse_csv_list(args.models, allowed=ALL_MODELS)
    seeds = parse_seed_list(args.seeds)
    device = resolve_device(args.device)
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
    write_csv(rq3_summary_path(root), summarize_rq3_rows(all_rows))
    print(f"[rq3] wrote rows={rq3_rows_path(root)} summary={rq3_summary_path(root)}")
    return all_rows


def run_channel_attribution(args: argparse.Namespace) -> list[dict[str, Any]]:
    spec = spec_from_args(args)
    root = Path(args.output_root)
    data_file = save_dataset(root, spec, overwrite=False)
    loaded_spec, splits = load_dataset(data_file)
    tasks = parse_csv_list(args.tasks, allowed=TASKS)
    seeds = parse_seed_list(args.seeds)
    device = resolve_device(args.device)
    rows: list[dict[str, Any]] = []
    for task in tasks:
        for seed in seeds:
            model, _, _ = load_trained_model(root, task, "gt_full", seed, device=device)
            rows.extend(
                channel_attribution_for_model(
                    model=model,
                    task=task,
                    split=splits["test"],
                    spec=loaded_spec,
                    seed=seed,
                    device=device,
                    resamples_per_site=args.resamples_per_site,
                    batch_size=args.eval_batch_size,
                    rq3_seed=args.rq3_seed,
                    min_oracle_effect=args.min_oracle_effect,
                )
            )
    write_csv(channel_rows_path(root), rows)
    write_csv(channel_summary_path(root), summarize_channel_rows(rows))
    print(f"[channel] wrote rows={channel_rows_path(root)} summary={channel_summary_path(root)}")
    return rows


def plot_accuracy_ladder(root: Path) -> Path:
    rows = read_csv(clean_metrics_path(root))
    agg = aggregate_mean_sem(rows, "poster_score", ("task", "model"))
    task_order = [task for task in TASKS if any(row["task"] == task for row in rows)]
    model_order = [model for model in DEFAULT_MODELS if any(row["model"] == model for row in rows)]
    colors = model_colors()
    fig, ax = plt.subplots(figsize=(10.6, 4.9))
    width = min(0.15, 0.78 / max(len(model_order), 1))
    x = np.arange(len(task_order), dtype=np.float64)
    for midx, model_name in enumerate(model_order):
        offsets = x + (midx - (len(model_order) - 1) / 2.0) * width
        means = [agg.get((task, model_name), {}).get("mean", np.nan) for task in task_order]
        sems = [agg.get((task, model_name), {}).get("sem", 0.0) for task in task_order]
        ax.bar(offsets, means, width=width, yerr=sems, capsize=3, color=colors[model_name], edgecolor="black", linewidth=0.5, label=MODEL_LABELS[model_name])
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABELS[task] for task in task_order])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Performance score (1 - MAE / target range)")
    ax.set_title("Accuracy Ladder: Range, Order, and Positional Identity")
    ax.legend(ncol=min(len(model_order), 5), frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    fig.tight_layout()
    path = figures_dir(root) / "fig1_accuracy_ladder.pdf"
    ensure_dir(path.parent)
    fig.savefig(path)
    plt.close(fig)
    print(f"[plot] wrote {path}")
    return path


def plot_channel_attribution(root: Path) -> Path:
    rows = read_csv(channel_summary_path(root))
    agg = aggregate_mean_sem(rows, "abs_survival_mean", ("task", "clamp"))
    task_order = [task for task in TASKS if any(row["task"] == task for row in rows)]
    clamps = ["Vs-clean-clamp", "B-mean-clamp"]
    colors = {"Vs-clean-clamp": "#d62728", "B-mean-clamp": "#1f77b4"}
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    width = 0.28
    x = np.arange(len(task_order), dtype=np.float64)
    for cidx, clamp in enumerate(clamps):
        offsets = x + (cidx - 0.5) * width
        means = [agg.get((task, clamp), {}).get("mean", np.nan) for task in task_order]
        sems = [agg.get((task, clamp), {}).get("sem", 0.0) for task in task_order]
        ax.bar(offsets, means, width=width, yerr=sems, capsize=3, color=colors[clamp], edgecolor="black", linewidth=0.5, label=clamp)
    ax.axhline(1.0, color="#666666", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABELS[task] for task in task_order])
    ax.set_ylabel("Fraction of GT-full effect surviving clamp")
    ax.set_title("Channel Attribution: Structural Value vs Routing Bias")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    fig.tight_layout()
    path = figures_dir(root) / "fig2_channel_attribution.pdf"
    ensure_dir(path.parent)
    fig.savefig(path)
    plt.close(fig)
    print(f"[plot] wrote {path}")
    return path


def plot_interaction_support(root: Path) -> Path:
    rows = [row for row in read_csv(rq3_rows_path(root)) if row["task"] in {"prod", "gap"}]
    summary = [row for row in read_csv(rq3_summary_path(root)) if row["task"] in {"prod", "gap"}]
    task_order = [task for task in ("prod", "gap") if any(row["task"] == task for row in rows)]
    model_order = [model for model in DEFAULT_MODELS if any(row["model"] == model for row in rows)]
    if not task_order or not model_order:
        raise ValueError("interaction support plot requires PROD or GAP RQ3 rows")
    colors = model_colors()
    fig, axes = plt.subplots(2, len(task_order), figsize=(5.8 * len(task_order), 7.0), squeeze=False, gridspec_kw={"height_ratios": [2.2, 1.0]})
    for col, task in enumerate(task_order):
        ax = axes[0, col]
        data: list[np.ndarray] = []
        labels: list[str] = []
        oracle_vals = np.asarray([float(row["interaction_oracle"]) for row in rows if row["task"] == task], dtype=np.float64)
        data.append(sample_for_plot(oracle_vals, 8000, 19 + col))
        labels.append("Oracle")
        for model_name in model_order:
            vals = np.asarray([float(row["interaction_model"]) for row in rows if row["task"] == task and row["model"] == model_name], dtype=np.float64)
            data.append(sample_for_plot(vals, 8000, 101 + col))
            labels.append(MODEL_LABELS[model_name])
        parts = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False)
        for idx, body in enumerate(parts["bodies"]):
            body.set_facecolor("#111111" if idx == 0 else colors[model_order[idx - 1]])
            body.set_alpha(0.76)
            body.set_edgecolor("black")
            body.set_linewidth(0.5)
        for idx, vals in enumerate(data, start=1):
            q1, med, q3 = np.percentile(vals, [25, 50, 75])
            ax.plot([idx - 0.18, idx + 0.18], [med, med], color="black", linewidth=1.2)
            ax.plot([idx, idx], [q1, q3], color="black", linewidth=2.0)
        ax.axhline(0.0, color="#555555", linestyle="--", linewidth=0.8)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_title(f"{TASK_LABELS[task]} RQ3 interaction")
        ax.set_ylabel("interaction = delta_AB - (delta_A + delta_B)")
        ax.grid(axis="y", color="#e0e0e0", linewidth=0.6)
        ax = axes[1, col]
        xs = np.arange(len(model_order), dtype=np.float64)
        corr_rows = {(row["task"], row["model"]): row for row in summary}
        vals = []
        for model_name in model_order:
            raw = corr_rows.get((task, model_name), {}).get("interaction_pearson", "nan")
            vals.append(float(raw) if raw not in {"", "nan"} else float("nan"))
        ax.bar(xs, vals, color=[colors[name] for name in model_order], edgecolor="black", linewidth=0.5)
        for xpos, val in zip(xs, vals):
            if math.isnan(float(val)):
                ax.text(xpos, 0.02, "no\nsignal", ha="center", va="bottom", fontsize=8, color="#444444")
        ax.set_xticks(xs)
        ax.set_xticklabels([MODEL_LABELS[name] for name in model_order], rotation=20, ha="right")
        ax.set_ylim(-1.05, 1.05)
        ax.set_ylabel("corr(model, oracle)")
        ax.set_title(f"{TASK_LABELS[task]} interaction-oracle correlation")
        ax.axhline(0.0, color="#555555", linewidth=0.8)
        ax.grid(axis="y", color="#e0e0e0", linewidth=0.6)
    fig.suptitle("Support: Pairwise Interaction with Oracle on PROD and GAP", y=0.995)
    fig.tight_layout()
    path = figures_dir(root) / "fig3_interaction_oracle_support.pdf"
    ensure_dir(path.parent)
    fig.savefig(path)
    plt.close(fig)
    print(f"[plot] wrote {path}")
    return path


def plot_add_calibration(root: Path) -> Path:
    rows = [row for row in read_csv(rq3_rows_path(root)) if row["task"] == "add"]
    model_order = [model for model in DEFAULT_MODELS if any(row["model"] == model for row in rows)]
    if not rows or not model_order:
        raise ValueError("ADD calibration plot requires ADD RQ3 rows")
    colors = model_colors()
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    data = []
    labels = []
    for model_name in model_order:
        vals = np.asarray([float(row["interaction_model"]) for row in rows if row["model"] == model_name], dtype=np.float64)
        data.append(sample_for_plot(vals, 8000, 301))
        labels.append(MODEL_LABELS[model_name])
    parts = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False)
    for body, model_name in zip(parts["bodies"], model_order):
        body.set_facecolor(colors[model_name])
        body.set_alpha(0.78)
        body.set_edgecolor("black")
        body.set_linewidth(0.5)
    for idx, vals in enumerate(data, start=1):
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        ax.plot([idx - 0.18, idx + 0.18], [med, med], color="black", linewidth=1.2)
        ax.plot([idx, idx], [q1, q3], color="black", linewidth=2.0)
    ax.axhline(0.0, color="#555555", linestyle="--", linewidth=0.9)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("ADD interaction")
    ax.set_title("Appendix: ADD Calibration and On-Manifold Continuous Resamples")
    ax.text(0.01, 0.98, "Oracle ADD interaction is exactly zero; resamples are Uniform[0,1].", transform=ax.transAxes, va="top", fontsize=9)
    ax.grid(axis="y", color="#e0e0e0", linewidth=0.6)
    fig.tight_layout()
    path = figures_dir(root) / "fig4_add_calibration_on_manifold.pdf"
    ensure_dir(path.parent)
    fig.savefig(path)
    plt.close(fig)
    print(f"[plot] wrote {path}")
    return path


def plot_schematic(root: Path, spec: ExperimentSpec) -> Path:
    fig, ax = plt.subplots(figsize=(10.8, 3.0))
    x = np.arange(spec.n_nodes)
    ax.plot(x, np.zeros_like(x), color="#888888", linewidth=1.4, zorder=1)
    ax.scatter(x, np.zeros_like(x), s=42, color="#d9d9d9", edgecolor="#555555", linewidth=0.5, zorder=2)
    ax.scatter([0], [0], s=130, color="#111111", edgecolor="black", zorder=4)
    ax.text(0, 0.19, "t = 0\nreadout", ha="center", va="bottom", fontsize=10, weight="bold")
    ax.axvspan(0.5, spec.receptive_radius + 0.5, color="#f2f2f2", alpha=1.0, zorder=0)
    ax.text(spec.receptive_radius / 2.0, -0.24, "MPNN radius r", ha="center", va="top", fontsize=9)
    far_start = spec.position_low
    ax.axvspan(far_start - 0.5, spec.position_high + 0.5, color="#fff2cc", alpha=0.7, zorder=0)
    example_p1 = far_start + 4
    example_p2 = min(spec.n_nodes - 2, far_start + 14)
    ax.scatter([example_p1, example_p2], [0, 0], s=155, color="#f58518", edgecolor="black", zorder=5)
    ax.text(example_p1, 0.21, "p1\nx1", ha="center", va="bottom", fontsize=10)
    ax.text(example_p2, 0.21, "p2\nx2", ha="center", va="bottom", fontsize=10)
    ax.annotate(
        "p1 < p2 sampled from [r+2, N-1]\ncontinuous values, no role flags",
        xy=((example_p1 + example_p2) / 2, 0),
        xytext=((example_p1 + example_p2) / 2, -0.57),
        ha="center",
        va="top",
        arrowprops={"arrowstyle": "-|>", "color": "#555555", "lw": 0.8},
        fontsize=10,
    )
    ax.text(
        spec.n_nodes - 1,
        0.21,
        "ADD = x1 + x2\nPROD = x1 * x2\nGAP = max(x1 - x2, 0)",
        ha="right",
        va="bottom",
        fontsize=10.5,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#bbbbbb"},
    )
    ax.set_xlim(-0.8, spec.n_nodes - 0.2)
    ax.set_ylim(-0.78, 0.58)
    ax.set_yticks([])
    ax.set_xlabel("path position / distance from readout endpoint")
    ax.set_title("Appendix: Minimal Path Setup")
    for spine in ("left", "right", "top"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    path = figures_dir(root) / "fig5_schematic.pdf"
    ensure_dir(path.parent)
    fig.savefig(path)
    plt.close(fig)
    print(f"[plot] wrote {path}")
    return path


def plot_all(args: argparse.Namespace) -> None:
    spec = spec_from_args(args)
    root = Path(args.output_root)
    plot_accuracy_ladder(root)
    if channel_summary_path(root).exists():
        plot_channel_attribution(root)
    rq3_rows = read_csv(rq3_rows_path(root)) if rq3_rows_path(root).exists() else []
    if any(row["task"] in {"prod", "gap"} for row in rq3_rows):
        plot_interaction_support(root)
    if any(row["task"] == "add" for row in rq3_rows):
        plot_add_calibration(root)
    plot_schematic(root, spec)


def model_colors() -> dict[str, str]:
    return {
        "mpnn": "#8c8c8c",
        "mpnn_vn": "#4c78a8",
        "nope_gt": "#54a24b",
        "gt_route": "#f58518",
        "gt_full": "#d62728",
        "gt_pe_anchor": "#9467bd",
    }


def sample_for_plot(values: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if values.size <= max_points:
        return values
    rng = np.random.default_rng(seed)
    return values[rng.choice(values.size, size=max_points, replace=False)]


def spec_from_args(args: argparse.Namespace) -> ExperimentSpec:
    return ExperimentSpec(
        n_nodes=int(args.n_nodes),
        receptive_radius=int(args.receptive_radius),
        train_size=int(args.train_size),
        val_size=int(args.val_size),
        test_size=int(args.test_size),
        data_seed=int(args.data_seed),
    )


def resolve_device(device_arg: str) -> torch.device:
    return torch.device(device_arg if device_arg != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))


def build_data_command(args: argparse.Namespace) -> None:
    spec = spec_from_args(args)
    save_dataset(Path(args.output_root), spec, overwrite=args.overwrite_data)
    write_json(
        config_record_path(Path(args.output_root)),
        {
            "spec": asdict(spec),
            "models": parse_csv_list(args.models, allowed=ALL_MODELS),
            "tasks": parse_csv_list(args.tasks, allowed=TASKS),
            "seeds": parse_seed_list(args.seeds),
            "hidden_dim": int(args.hidden_dim),
            "gt_heads": int(args.gt_heads),
            "resamples_per_site": int(args.resamples_per_site),
            "gt_route_vs_gt_full": {
                "same_routing_bias": True,
                "gt_route_value_transport": "content-only",
                "gt_full_value_transport": "content + x_value * learned_position_embedding",
                "routing_clamp": "distance-bias replaced by per-head mean",
                "value_clamp": "source Vs replaced by clean Vs",
            },
        },
    )


def run_all(args: argparse.Namespace) -> None:
    build_data_command(args)
    run_training(args)
    clean_evaluate(args)
    run_rq3(args)
    requested_models = set(parse_csv_list(args.models, allowed=ALL_MODELS))
    if args.run_channel_attribution and "gt_full" in requested_models:
        run_channel_attribution(args)
    elif args.run_channel_attribution:
        print("[channel] skipped because requested models do not include gt_full")
    plot_all(args)
    print(f"[done] minimal v2 experiment complete: {Path(args.output_root)}")


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
        f"--hidden-dim {args.hidden_dim} --gt-heads {args.gt_heads} "
        f"--max-epochs {args.max_epochs} --patience {args.patience} "
        f"--batch-size {args.batch_size} --eval-batch-size {args.eval_batch_size} "
        f"--resamples-per-site {args.resamples_per_site} --log-every {args.log_every} "
        f"--min-oracle-effect {args.min_oracle_effect}"
    )
    if args.run_channel_attribution:
        flags += " --run-channel-attribution"
    wrap = common + module + " run-all " + flags + " --device cuda"
    print("mkdir -p logs")
    print(
        "sbatch -A mlmi-jgg45-sl2-gpu -p ampere --qos=gpu1 "
        "--gres=gpu:1 --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=80G --time=01:00:00 "
        "-J min-order-v2 -o logs/min-order-v2-%j.out -e logs/min-order-v2-%j.err "
        f"--wrap {json.dumps(wrap)}"
    )


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-root", type=Path, default=default_output_root())
    parser.add_argument("--tasks", type=str, default="add,prod,gap")
    parser.add_argument("--models", type=str, default="mpnn,mpnn_vn,nope_gt,gt_route,gt_full")
    parser.add_argument("--seeds", type=str, default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--n-nodes", type=int, default=32)
    parser.add_argument("--receptive-radius", type=int, default=4)
    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--data-seed", type=int, default=7001)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--gt-heads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--max-epochs", type=int, default=800)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--resamples-per-site", type=int, default=4)
    parser.add_argument("--rq3-seed", type=int, default=8101)
    parser.add_argument("--min-oracle-effect", type=float, default=1.0e-4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--run-channel-attribution", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite-data", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--overwrite-checkpoints", action=argparse.BooleanOptionalAction, default=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, help_text, fn in [
        ("build-data", "Generate and cache the continuous path dataset.", build_data_command),
        ("train", "Train selected models/tasks/seeds.", run_training),
        ("evaluate", "Evaluate clean held-out regression performance.", clean_evaluate),
        ("run-rq3", "Run the output-level double-resample interaction test.", run_rq3),
        ("run-channel-attribution", "Clamp GT-full structural value/routing channels.", run_channel_attribution),
        ("plot", "Generate all poster figures from cached metrics.", plot_all),
        ("run-all", "Run data, training, clean eval, RQ3, channel attribution, and figures.", run_all),
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
