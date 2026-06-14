"""Minimal experiment v3: four-axis long-range architecture decomposition.

This runner implements ``minimal_experiment_v3.md`` as a standalone synthetic
experiment.  It trains a matched ladder of local, symmetric-global, content-only
dense, routing-structured, and full structural-value models on continuous path
tasks, then runs the RQ3 interaction, RELMODE rank sweep, bounded channel
ablation, source-map, RQ0, and RQ4 validation figures.
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


TASKS = ("add", "prod", "gap", "relmode")
TASK_LABELS = {"add": "ADD", "prod": "PROD", "gap": "GAP", "relmode": "RELMODE"}
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
    mode_count: int = 6
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
        return {"add": 2.0, "prod": 1.0, "gap": 1.0, "relmode": 4.0}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_output_root() -> Path:
    return Path.cwd() / "artifacts" / "minimal_order_dissociation_v3"


def data_path(root: Path, spec: ExperimentSpec) -> Path:
    return (
        root
        / "data"
        / f"path_identity_v3_N{spec.n_nodes}_r{spec.receptive_radius}_K{spec.mode_count}_seed{spec.data_seed}.pt"
    )


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


def k_sweep_metrics_path(root: Path) -> Path:
    return metrics_dir(root) / "relmode_k_sweep_clean_performance.csv"


def rq0_profile_path(root: Path) -> Path:
    return metrics_dir(root) / "rq0_demand_profile.csv"


def source_map_path(root: Path) -> Path:
    return metrics_dir(root) / "source_map_recovery.csv"


def rq4_address_path(root: Path) -> Path:
    return metrics_dir(root) / "rq4_address_mode.csv"


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


def parse_int_list(text: str) -> list[int]:
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


def relation_bucket_index(position: torch.Tensor, spec: ExperimentSpec) -> torch.Tensor:
    span = max(1, spec.position_high - spec.position_low + 1)
    rel = (position.long() - int(spec.position_low)).clamp(min=0, max=span - 1)
    bucket = torch.div(rel * int(spec.mode_count), span, rounding_mode="floor")
    return bucket.clamp(min=0, max=int(spec.mode_count) - 1).long()


def relmode_phi(bucket: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    return torch.cos((bucket.float() + 1.0) * math.pi * value.float())


def oracle_scalar(
    task: str,
    x1: torch.Tensor,
    x2: torch.Tensor,
    spec: ExperimentSpec | None = None,
    p1: torch.Tensor | None = None,
    p2: torch.Tensor | None = None,
) -> torch.Tensor:
    if task == "add":
        return x1.float() + x2.float()
    if task == "prod":
        return x1.float() * x2.float()
    if task == "gap":
        return torch.clamp(x1.float() - x2.float(), min=0.0)
    if task == "relmode":
        if spec is None or p1 is None or p2 is None:
            raise ValueError("RELMODE oracle requires spec, p1, and p2")
        r1 = relation_bucket_index(p1, spec).to(x1.device)
        r2 = relation_bucket_index(p2, spec).to(x2.device)
        return relmode_phi(r1, x1) + relmode_phi(r2, x2)
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
        "r1": relation_bucket_index(p1_t, spec),
        "r2": relation_bucket_index(p2_t, spec),
        "x1": x1_t,
        "x2": x2_t,
        "y_add": oracle_scalar("add", x1_t, x2_t),
        "y_prod": oracle_scalar("prod", x1_t, x2_t),
        "y_gap": oracle_scalar("gap", x1_t, x2_t),
        "y_relmode": oracle_scalar("relmode", x1_t, x2_t, spec, p1_t, p2_t),
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
        if not torch.equal(split["r1"], relation_bucket_index(p1, spec)):
            raise ValueError(f"{split_name}: p1 relation buckets are inconsistent")
        if not torch.equal(split["r2"], relation_bucket_index(p2, spec)):
            raise ValueError(f"{split_name}: p2 relation buckets are inconsistent")
        if not torch.allclose(split["y_add"], oracle_scalar("add", x1, x2)):
            raise ValueError(f"{split_name}: ADD labels are inconsistent")
        if not torch.allclose(split["y_prod"], oracle_scalar("prod", x1, x2)):
            raise ValueError(f"{split_name}: PROD labels are inconsistent")
        if not torch.allclose(split["y_gap"], oracle_scalar("gap", x1, x2)):
            raise ValueError(f"{split_name}: GAP labels are inconsistent")
        if not torch.allclose(split["y_relmode"], oracle_scalar("relmode", x1, x2, spec, p1, p2)):
            raise ValueError(f"{split_name}: RELMODE labels are inconsistent")
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
        f"test={spec.test_size} N={spec.n_nodes} r={spec.receptive_radius} K={spec.mode_count}"
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
    GT-full keeps the same routing and feeds a relation/position embedding into
    the value MLP so the transported message can compute relation-conditioned
    transforms such as RELMODE.
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
        self.value_content = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.struct_value = nn.Embedding(n_nodes, hidden_dim)
        self.value_structured = nn.Sequential(
            nn.Linear(in_dim + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
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

    def routing_bias_channel(self, *, device: torch.device, dtype: torch.dtype, ablate_b: bool = False) -> torch.Tensor:
        if not self.use_routing_bias:
            return torch.zeros(self.num_heads, self.n_nodes, device=device, dtype=dtype)
        if ablate_b:
            return torch.zeros(self.num_heads, self.n_nodes, device=device, dtype=dtype)
        bias = self.distance_bias(self.receiver_dist.to(device)).transpose(0, 1).to(dtype=dtype)
        return bias

    def value_struct_channel(self, x: torch.Tensor) -> torch.Tensor:
        pos = self.positions.to(x.device)
        struct = self.struct_value(pos)[None, :, :].to(dtype=x.dtype)
        marker = x[:, :, 0:1]
        return marker * struct

    def value_channel(self, x: torch.Tensor, *, ablate_vs: bool = False) -> torch.Tensor:
        if not self.use_structural_value or ablate_vs:
            return self.value_content(x)
        return self.value_structured(torch.cat([x, self.value_struct_channel(x)], dim=-1))

    def forward(
        self,
        x: torch.Tensor,
        *,
        ablate_vs: bool = False,
        ablate_b: bool = False,
    ) -> torch.Tensor:
        h = self.content(x)
        bsz, n_nodes, hidden = h.shape
        k = self.key(h).view(bsz, n_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value_channel(x, ablate_vs=ablate_vs).view(bsz, n_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.einsum("hd,bhnd->bhn", self.query.to(x.device), k) / math.sqrt(self.head_dim)
        scores = scores + self.routing_bias_channel(device=x.device, dtype=scores.dtype, ablate_b=ablate_b)[None, :, :]
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
    forward_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    model.eval()
    forward_kwargs = dict(forward_kwargs or {})
    losses: list[float] = []
    errors: list[torch.Tensor] = []
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    n = int(split["x"].shape[0])
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = {key: value[start:end].to(device) for key, value in split.items()}
        output = model(batch["x"], **forward_kwargs).reshape(-1)
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
    score = max(0.0, min(1.0, 1.0 - mae / max(spec.label_range[task], EPS)))
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
            y_base = oracle_scalar(task, x1, x2, spec, p1, p2)
            y_a = oracle_scalar(task, x1_new, x2, spec, p1, p2)
            y_b = oracle_scalar(task, x1, x2_new, spec, p1, p2)
            y_ab = oracle_scalar(task, x1_new, x2_new, spec, p1, p2)
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
                        "r1": int(relation_bucket_index(p1[local_idx : local_idx + 1], spec)[0].item()),
                        "r2": int(relation_bucket_index(p2[local_idx : local_idx + 1], spec)[0].item()),
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
        corr = (
            float("nan")
            if oracle_std < INTERACTION_SIGNAL_EPS
            else pearson(model, oracle, min_std=INTERACTION_SIGNAL_EPS)
        )
        if oracle_std < INTERACTION_SIGNAL_EPS:
            status = "not_applicable_zero_oracle"
        else:
            status = "ok" if model_std >= INTERACTION_SIGNAL_EPS else "no_model_signal"
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
    batch_size: int,
) -> list[dict[str, Any]]:
    if not isinstance(model, StructuralTransportReadout) or not model.use_structural_value:
        raise TypeError("channel attribution requires GT-full StructuralTransportReadout")
    rows: list[dict[str, Any]] = []
    ablations = [
        ("clean", {}),
        ("Vs-ablated", {"ablate_vs": True}),
        ("B-ablated", {"ablate_b": True}),
    ]
    for ablation, kwargs in ablations:
        metrics = evaluate_model(
            model,
            task,
            split,
            spec,
            device=device,
            batch_size=batch_size,
            forward_kwargs=kwargs,
        )
        rows.append(
            {
                "task": task,
                "task_label": TASK_LABELS[task],
                "model": "gt_full",
                "model_label": MODEL_LABELS["gt_full"],
                "seed": int(seed),
                "ablation": ablation,
                "test_loss": metrics["loss"],
                "test_mae": metrics["mae"],
                "test_mse": metrics["mse"],
                "test_r2": metrics["r2"],
                "poster_score": metrics["score"],
                "n_test": int(split["x"].shape[0]),
            }
        )
    return rows


def summarize_channel_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["task"]), str(row["ablation"])), []).append(row)
    summary: list[dict[str, Any]] = []
    for (task, ablation), group in sorted(grouped.items()):
        scores = np.asarray([float(row["poster_score"]) for row in group], dtype=np.float64)
        maes = np.asarray([float(row["test_mae"]) for row in group], dtype=np.float64)
        r2s = np.asarray([float(row["test_r2"]) for row in group], dtype=np.float64)
        summary.append(
            {
                "task": task,
                "task_label": TASK_LABELS[task],
                "model": "gt_full",
                "model_label": MODEL_LABELS["gt_full"],
                "ablation": ablation,
                "n_seeds": int(len(group)),
                "poster_score_mean": float(np.mean(scores)) if scores.size else float("nan"),
                "poster_score_sem": float(np.std(scores, ddof=1) / math.sqrt(scores.size)) if scores.size > 1 else 0.0,
                "test_mae_mean": float(np.mean(maes)) if maes.size else float("nan"),
                "test_r2_mean": float(np.mean(r2s)) if r2s.size else float("nan"),
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
                    batch_size=args.eval_batch_size,
                )
            )
    write_csv(channel_rows_path(root), rows)
    write_csv(channel_summary_path(root), summarize_channel_rows(rows))
    print(f"[channel] wrote rows={channel_rows_path(root)} summary={channel_summary_path(root)}")
    return rows


def run_rq0_demand_profile(args: argparse.Namespace) -> list[dict[str, Any]]:
    spec = spec_from_args(args)
    root = Path(args.output_root)
    data_file = save_dataset(root, spec, overwrite=False)
    loaded_spec, splits = load_dataset(data_file)
    tasks = parse_csv_list(args.tasks, allowed=TASKS)
    split = splits["test"]
    rng = torch.Generator().manual_seed(int(args.rq3_seed) + 515)
    rows: list[dict[str, Any]] = []
    n = min(int(args.methodology_graphs), int(split["x"].shape[0]))
    for idx in range(n):
        p1 = split["p1"][idx : idx + 1]
        p2 = split["p2"][idx : idx + 1]
        x1 = split["x1"][idx : idx + 1]
        x2 = split["x2"][idx : idx + 1]
        x1_new = torch.rand(1, generator=rng)
        x2_new = torch.rand(1, generator=rng)
        for task in tasks:
            base = oracle_scalar(task, x1, x2, loaded_spec, p1, p2)
            y_a = oracle_scalar(task, x1_new, x2, loaded_spec, p1, p2)
            y_b = oracle_scalar(task, x1, x2_new, loaded_spec, p1, p2)
            rows.extend(
                [
                    {
                        "task": task,
                        "task_label": TASK_LABELS[task],
                        "sample_id": idx,
                        "site": "p1",
                        "distance": int(p1.item()),
                        "oracle_abs_delta": float(torch.abs(y_a - base).item()),
                    },
                    {
                        "task": task,
                        "task_label": TASK_LABELS[task],
                        "sample_id": idx,
                        "site": "p2",
                        "distance": int(p2.item()),
                        "oracle_abs_delta": float(torch.abs(y_b - base).item()),
                    },
                ]
            )
    write_csv(rq0_profile_path(root), rows)
    print(f"[rq0] wrote {rq0_profile_path(root)}")
    return rows


@torch.no_grad()
def source_effect_map(model: nn.Module, x_base: torch.Tensor) -> torch.Tensor:
    n_nodes = int(x_base.shape[1])
    base = model(x_base).reshape(1)
    x_erased = x_base.repeat(n_nodes, 1, 1)
    idx = torch.arange(n_nodes, device=x_base.device)
    x_erased[idx, idx, :] = 0.0
    pred = model(x_erased).reshape(-1)
    return torch.abs(pred - base)


def source_map_rows_for(
    *,
    model: nn.Module,
    task: str,
    model_name: str,
    seed: int,
    split: Mapping[str, torch.Tensor],
    spec: ExperimentSpec,
    device: torch.device,
    graph_limit: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n = min(int(graph_limit), int(split["x"].shape[0]))
    for idx in range(n):
        x_base = split["x"][idx : idx + 1].to(device)
        effects = source_effect_map(model, x_base).detach().cpu()
        p1 = int(split["p1"][idx].item())
        p2 = int(split["p2"][idx].item())
        order = torch.argsort(effects, descending=True)
        ranks = torch.empty_like(order)
        ranks[order] = torch.arange(len(order))
        for node in range(spec.n_nodes):
            rows.append(
                {
                    "task": task,
                    "task_label": TASK_LABELS[task],
                    "model": model_name,
                    "model_label": MODEL_LABELS[model_name],
                    "seed": int(seed),
                    "sample_id": idx,
                    "node": node,
                    "p1": p1,
                    "p2": p2,
                    "is_source": node in {p1, p2},
                    "source_effect": float(effects[node].item()),
                    "effect_rank": int(ranks[node].item()) + 1,
                }
            )
    return rows


def run_source_maps(args: argparse.Namespace) -> list[dict[str, Any]]:
    spec = spec_from_args(args)
    root = Path(args.output_root)
    data_file = save_dataset(root, spec, overwrite=False)
    loaded_spec, splits = load_dataset(data_file)
    seeds = parse_seed_list(args.seeds)
    device = resolve_device(args.device)
    pairs = [("gap", "gt_full"), ("relmode", "gt_route"), ("relmode", "gt_full")]
    rows: list[dict[str, Any]] = []
    for task, model_name in pairs:
        for seed in seeds:
            ckpt = checkpoint_path(root, task, model_name, seed)
            if not ckpt.exists():
                print(f"[source-map] skip missing checkpoint {ckpt}")
                continue
            model, _, _ = load_trained_model(root, task, model_name, seed, device=device)
            rows.extend(
                source_map_rows_for(
                    model=model,
                    task=task,
                    model_name=model_name,
                    seed=seed,
                    split=splits["test"],
                    spec=loaded_spec,
                    device=device,
                    graph_limit=args.methodology_graphs,
                )
            )
    write_csv(source_map_path(root), rows)
    print(f"[source-map] wrote {source_map_path(root)}")
    return rows


def run_rq4_address_mode(args: argparse.Namespace) -> list[dict[str, Any]]:
    spec = spec_from_args(args)
    root = Path(args.output_root)
    data_file = save_dataset(root, spec, overwrite=False)
    loaded_spec, splits = load_dataset(data_file)
    seeds = parse_seed_list(args.seeds)
    device = resolve_device(args.device)
    rng = random.Random(int(args.rq3_seed) + 919)
    rows: list[dict[str, Any]] = []
    task = "gap"
    model_name = "gt_full"
    for seed in seeds:
        ckpt = checkpoint_path(root, task, model_name, seed)
        if not ckpt.exists():
            print(f"[rq4] skip missing checkpoint {ckpt}")
            continue
        model, _, _ = load_trained_model(root, task, model_name, seed, device=device)
        n = min(int(args.methodology_graphs), int(splits["test"]["x"].shape[0]))
        candidates = list(range(loaded_spec.position_low, loaded_spec.position_high + 1))
        for idx in range(n):
            x_base = splits["test"]["x"][idx : idx + 1].to(device)
            p1 = int(splits["test"]["p1"][idx].item())
            p2 = int(splits["test"]["p2"][idx].item())
            available = [pos for pos in candidates if pos not in {p1, p2}]
            q1, q2 = sorted(rng.sample(available, 2))
            x_shuffle = x_base.clone()
            x_shuffle[:, [p1, p2, q1, q2], :] = 0.0
            x_shuffle[:, q1, :] = x_base[:, p1, :]
            x_shuffle[:, q2, :] = x_base[:, p2, :]
            for condition, x_cur, old_nodes, new_nodes in [
                ("clean", x_base, {p1, p2}, {q1, q2}),
                ("content_shuffled", x_shuffle, {p1, p2}, {q1, q2}),
            ]:
                effects = source_effect_map(model, x_cur).detach().cpu()
                old_effect = float(torch.mean(effects[list(old_nodes)]).item())
                new_effect = float(torch.mean(effects[list(new_nodes)]).item())
                rows.append(
                    {
                        "task": task,
                        "task_label": TASK_LABELS[task],
                        "model": model_name,
                        "model_label": MODEL_LABELS[model_name],
                        "seed": int(seed),
                        "sample_id": idx,
                        "condition": condition,
                        "old_p1": p1,
                        "old_p2": p2,
                        "new_p1": q1,
                        "new_p2": q2,
                        "old_source_effect_mean": old_effect,
                        "new_source_effect_mean": new_effect,
                        "content_follow_score": new_effect - old_effect,
                    }
                )
    write_csv(rq4_address_path(root), rows)
    print(f"[rq4] wrote {rq4_address_path(root)}")
    return rows


def run_k_sweep(args: argparse.Namespace) -> list[dict[str, Any]]:
    base_spec = spec_from_args(args)
    root = Path(args.output_root)
    k_values = parse_int_list(args.k_sweep_values)
    models = parse_csv_list(args.k_sweep_models, allowed=ALL_MODELS)
    seeds = parse_seed_list(args.seeds)
    device = resolve_device(args.device)
    rows: list[dict[str, Any]] = []
    for k in k_values:
        sweep_spec = ExperimentSpec(
            n_nodes=base_spec.n_nodes,
            receptive_radius=base_spec.receptive_radius,
            mode_count=int(k),
            train_size=int(args.k_sweep_train_size or base_spec.train_size),
            val_size=base_spec.val_size,
            test_size=base_spec.test_size,
            data_seed=base_spec.data_seed + int(k) * 31,
        )
        sweep_root = root / "k_sweep" / f"K{k}"
        data_file = save_dataset(sweep_root, sweep_spec, overwrite=args.overwrite_data)
        _, splits = load_dataset(data_file)
        for model_name in models:
            for seed in seeds:
                train_one(
                    root=sweep_root,
                    data_file=data_file,
                    task="relmode",
                    model_name=model_name,
                    seed=seed,
                    device=device,
                    hidden_dim=args.hidden_dim,
                    gt_heads=args.gt_heads,
                    batch_size=args.batch_size,
                    max_epochs=args.k_sweep_max_epochs or args.max_epochs,
                    patience=args.k_sweep_patience or args.patience,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    log_every=args.log_every,
                    overwrite=args.overwrite_checkpoints,
                )
                model, _, payload = load_trained_model(sweep_root, "relmode", model_name, seed, device=device)
                metrics = evaluate_model(model, "relmode", splits["test"], sweep_spec, device=device, batch_size=args.eval_batch_size)
                rows.append(
                    {
                        "K": int(k),
                        "head_budget_H": int(args.gt_heads),
                        "task": "relmode",
                        "task_label": TASK_LABELS["relmode"],
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
                    }
                )
                print(f"[k-sweep] K={k} model={model_name} seed={seed} score={metrics['score']:.5g}")
    write_csv(k_sweep_metrics_path(root), rows)
    print(f"[k-sweep] wrote {k_sweep_metrics_path(root)}")
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
    ax.set_title("Ingredient Ladder: Range, Order, Identity, and Relational Transport")
    ax.legend(ncol=min(len(model_order), 5), frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    fig.tight_layout()
    path = figures_dir(root) / "fig1_ingredient_ladder.pdf"
    ensure_dir(path.parent)
    fig.savefig(path)
    plt.close(fig)
    print(f"[plot] wrote {path}")
    return path


def plot_channel_attribution(root: Path) -> Path:
    rows = read_csv(channel_summary_path(root))
    agg = {(row["task"], row["ablation"]): row for row in rows}
    task_order = [task for task in TASKS if any(row["task"] == task for row in rows)]
    ablations = ["clean", "Vs-ablated", "B-ablated"]
    colors = {"clean": "#444444", "Vs-ablated": "#d62728", "B-ablated": "#1f77b4"}
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    width = 0.22
    x = np.arange(len(task_order), dtype=np.float64)
    for aidx, ablation in enumerate(ablations):
        offsets = x + (aidx - (len(ablations) - 1) / 2.0) * width
        means = [float(agg.get((task, ablation), {}).get("poster_score_mean", "nan")) for task in task_order]
        sems = [float(agg.get((task, ablation), {}).get("poster_score_sem", 0.0)) for task in task_order]
        ax.bar(offsets, means, width=width, yerr=sems, capsize=3, color=colors[ablation], edgecolor="black", linewidth=0.5, label=ablation)
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABELS[task] for task in task_order])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("GT-full performance score")
    ax.set_title("Channel Ablation: Clean vs Structural Value and Routing Bias Removed")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    fig.tight_layout()
    path = figures_dir(root) / "fig3_channel_ablation.pdf"
    ensure_dir(path.parent)
    fig.savefig(path)
    plt.close(fig)
    print(f"[plot] wrote {path}")
    return path


def plot_k_sweep(root: Path) -> Path:
    rows = read_csv(k_sweep_metrics_path(root))
    model_order = [model for model in DEFAULT_MODELS if any(row["model"] == model for row in rows)]
    colors = model_colors()
    grouped = aggregate_mean_sem(rows, "poster_score", ("K", "model"))
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for model_name in model_order:
        ks = sorted({int(row["K"]) for row in rows if row["model"] == model_name})
        means = [grouped.get((str(k), model_name), {}).get("mean", np.nan) for k in ks]
        sems = [grouped.get((str(k), model_name), {}).get("sem", 0.0) for k in ks]
        ax.errorbar(
            ks,
            means,
            yerr=sems,
            marker="o",
            linewidth=1.8,
            capsize=3,
            color=colors[model_name],
            label=MODEL_LABELS[model_name],
        )
    if rows:
        head_budget = int(float(rows[0]["head_budget_H"]))
        ax.axvline(head_budget, color="#555555", linestyle="--", linewidth=0.9)
        ax.text(head_budget + 0.05, 0.04, "H", fontsize=9, color="#444444")
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("RELMODE bucket count K")
    ax.set_ylabel("Performance score")
    ax.set_title("RELMODE Capacity Kink at Fixed Head Budget")
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    fig.tight_layout()
    path = figures_dir(root) / "fig2_relmode_capacity_kink.pdf"
    ensure_dir(path.parent)
    fig.savefig(path)
    plt.close(fig)
    print(f"[plot] wrote {path}")
    return path


def plot_interaction_support(root: Path) -> Path:
    rows = read_csv(rq3_rows_path(root))
    summary = read_csv(rq3_summary_path(root))
    task_order = [task for task in TASKS if any(row["task"] == task for row in rows)]
    model_order = [model for model in DEFAULT_MODELS if any(row["model"] == model for row in rows)]
    if not task_order or not model_order:
        raise ValueError("interaction support plot requires RQ3 rows")
    colors = model_colors()
    fig, axes = plt.subplots(2, len(task_order), figsize=(4.5 * len(task_order), 7.0), squeeze=False, gridspec_kw={"height_ratios": [2.2, 1.0]})
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
    fig.suptitle("Four-Task RQ3 Interaction: Order Readout and Its Limits", y=0.995)
    fig.tight_layout()
    path = figures_dir(root) / "fig5_four_task_interaction_panel.pdf"
    ensure_dir(path.parent)
    fig.savefig(path)
    plt.close(fig)
    print(f"[plot] wrote {path}")
    return path


def plot_source_map_recovery(root: Path) -> Path:
    rows = read_csv(source_map_path(root))
    if not rows:
        raise ValueError("source-map plot requires source-map rows")
    groups = sorted({(row["task"], row["model"]) for row in rows}, key=lambda x: (TASKS.index(x[0]), DEFAULT_MODELS.index(x[1]) if x[1] in DEFAULT_MODELS else 99))
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    width = 0.22
    x = np.arange(len(groups), dtype=np.float64)
    for idx, source_flag in enumerate([True, False]):
        label = "planted value nodes" if source_flag else "other nodes"
        vals = []
        sems = []
        for task, model_name in groups:
            by_seed: dict[int, list[float]] = {}
            for row in rows:
                if row["task"] == task and row["model"] == model_name and str(row["is_source"]).lower() == str(source_flag).lower():
                    by_seed.setdefault(int(row["seed"]), []).append(float(row["source_effect"]))
            seed_means = np.asarray([np.mean(v) for v in by_seed.values()], dtype=np.float64)
            vals.append(float(seed_means.mean()) if seed_means.size else np.nan)
            sems.append(float(seed_means.std(ddof=1) / math.sqrt(seed_means.size)) if seed_means.size > 1 else 0.0)
        ax.bar(x + (idx - 0.5) * width, vals, width=width, yerr=sems, capsize=3, color=("#f58518" if source_flag else "#bdbdbd"), edgecolor="black", linewidth=0.5, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{TASK_LABELS[t]}\n{MODEL_LABELS[m]}" for t, m in groups], rotation=0)
    ax.set_ylabel("Leave-node-out source effect")
    ax.set_title("Source-Map Recovery: Reached Sources vs Non-Sources")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    fig.tight_layout()
    path = figures_dir(root) / "fig4_source_map_recovery.pdf"
    ensure_dir(path.parent)
    fig.savefig(path)
    plt.close(fig)
    print(f"[plot] wrote {path}")
    return path


def plot_rq0_demand_profile(root: Path) -> Path:
    rows = read_csv(rq0_profile_path(root))
    if not rows:
        raise ValueError("RQ0 plot requires demand rows")
    task_order = [task for task in TASKS if any(row["task"] == task for row in rows)]
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    colors = {"add": "#4c78a8", "prod": "#54a24b", "gap": "#f58518", "relmode": "#d62728"}
    for task in task_order:
        by_distance: dict[int, list[float]] = {}
        for row in rows:
            if row["task"] == task:
                by_distance.setdefault(int(row["distance"]), []).append(float(row["oracle_abs_delta"]))
        xs = sorted(by_distance)
        ys = [float(np.mean(by_distance[d])) for d in xs]
        ax.plot(xs, ys, marker="o", linewidth=1.5, color=colors[task], label=TASK_LABELS[task])
    ax.set_xlabel("Distance from readout t")
    ax.set_ylabel("Mean oracle |delta| under single-site resample")
    ax.set_title("RQ0 Demand Profile: Uniform Long-Range Oracle Demand")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    fig.tight_layout()
    path = figures_dir(root) / "fig6_rq0_demand_profile.pdf"
    ensure_dir(path.parent)
    fig.savefig(path)
    plt.close(fig)
    print(f"[plot] wrote {path}")
    return path


def plot_rq4_address_mode(root: Path) -> Path:
    rows = read_csv(rq4_address_path(root))
    if not rows:
        raise ValueError("RQ4 plot requires address-mode rows")
    groups = ["clean", "content_shuffled"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    width = 0.3
    x = np.arange(len(groups), dtype=np.float64)
    for idx, key in enumerate(["old_source_effect_mean", "new_source_effect_mean"]):
        vals = []
        sems = []
        for condition in groups:
            by_seed: dict[int, list[float]] = {}
            for row in rows:
                if row["condition"] == condition:
                    by_seed.setdefault(int(row["seed"]), []).append(float(row[key]))
            seed_means = np.asarray([np.mean(v) for v in by_seed.values()], dtype=np.float64)
            vals.append(float(seed_means.mean()) if seed_means.size else np.nan)
            sems.append(float(seed_means.std(ddof=1) / math.sqrt(seed_means.size)) if seed_means.size > 1 else 0.0)
        label = "old positions" if key.startswith("old") else "content-moved positions"
        ax.bar(x + (idx - 0.5) * width, vals, width=width, yerr=sems, capsize=3, color=("#8c8c8c" if idx == 0 else "#f58518"), edgecolor="black", linewidth=0.5, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(["clean", "content shuffled"])
    ax.set_ylabel("Source-map effect")
    ax.set_title("RQ4 Address Mode: Source Map Follows Content")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    fig.tight_layout()
    path = figures_dir(root) / "fig7_rq4_address_mode.pdf"
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
    path = figures_dir(root) / "fig8_add_calibration_on_manifold.pdf"
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
        f"ADD = x1 + x2\nPROD = x1 * x2\nGAP = max(x1 - x2, 0)\nRELMODE = phi_r1(x1) + phi_r2(x2)\nK = {spec.mode_count} distance buckets",
        ha="right",
        va="bottom",
        fontsize=10.5,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#bbbbbb"},
    )
    ax.set_xlim(-0.8, spec.n_nodes - 0.2)
    ax.set_ylim(-0.78, 0.58)
    ax.set_yticks([])
    ax.set_xlabel("path position / distance from readout endpoint")
    ax.set_title("Appendix: Minimal Path Setup with RELMODE Buckets")
    for spine in ("left", "right", "top"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    path = figures_dir(root) / "fig9_schematic.pdf"
    ensure_dir(path.parent)
    fig.savefig(path)
    plt.close(fig)
    print(f"[plot] wrote {path}")
    return path


def plot_all(args: argparse.Namespace) -> None:
    spec = spec_from_args(args)
    root = Path(args.output_root)
    if clean_metrics_path(root).exists():
        plot_accuracy_ladder(root)
    if k_sweep_metrics_path(root).exists():
        plot_k_sweep(root)
    if channel_summary_path(root).exists():
        plot_channel_attribution(root)
    if source_map_path(root).exists():
        plot_source_map_recovery(root)
    rq3_rows = read_csv(rq3_rows_path(root)) if rq3_rows_path(root).exists() else []
    if rq3_rows:
        plot_interaction_support(root)
    if rq0_profile_path(root).exists():
        plot_rq0_demand_profile(root)
    if rq4_address_path(root).exists():
        plot_rq4_address_mode(root)
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
        mode_count=int(args.mode_count),
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
            "mode_count": int(args.mode_count),
            "mode_set": "phi_k(x)=cos((k+1)*pi*x)",
            "resamples_per_site": int(args.resamples_per_site),
            "gt_route_vs_gt_full": {
                "same_routing_bias": True,
                "gt_route_value_transport": "content-only",
                "gt_full_value_transport": "content plus learned structural value embedding concatenated into value MLP",
                "B_ablation": "zero structural attention bias; attention becomes content-only",
                "Vs_ablation": "remove structural value embedding; values become content-only",
                "accuracy_metric": "bounded score = clip(1 - MAE / target_range, 0, 1)",
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
    if args.run_methodology:
        run_rq0_demand_profile(args)
        run_source_maps(args)
        run_rq4_address_mode(args)
    if args.run_k_sweep:
        run_k_sweep(args)
    plot_all(args)
    print(f"[done] minimal v3 experiment complete: {Path(args.output_root)}")


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
        f"--mode-count {args.mode_count} --train-size {args.train_size} --val-size {args.val_size} "
        f"--test-size {args.test_size} --data-seed {args.data_seed} "
        f"--hidden-dim {args.hidden_dim} --gt-heads {args.gt_heads} "
        f"--max-epochs {args.max_epochs} --patience {args.patience} "
        f"--batch-size {args.batch_size} --eval-batch-size {args.eval_batch_size} "
        f"--lr {args.lr} --weight-decay {args.weight_decay} "
        f"--resamples-per-site {args.resamples_per_site} --rq3-seed {args.rq3_seed} --log-every {args.log_every} "
        f"--methodology-graphs {args.methodology_graphs} "
        f"--k-sweep-values {args.k_sweep_values} --k-sweep-models {args.k_sweep_models} "
        f"--k-sweep-train-size {args.k_sweep_train_size} "
        f"--k-sweep-max-epochs {args.k_sweep_max_epochs} --k-sweep-patience {args.k_sweep_patience}"
    )
    if args.run_channel_attribution:
        flags += " --run-channel-attribution"
    if args.run_methodology:
        flags += " --run-methodology"
    if args.run_k_sweep:
        flags += " --run-k-sweep"
    wrap = common + module + " run-all " + flags + " --device cuda"
    print("mkdir -p logs")
    print(
        "sbatch -A mlmi-jgg45-sl2-gpu -p ampere --qos=gpu1 "
        "--gres=gpu:1 --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=80G --time=01:00:00 "
        "-J min-order-v3 -o logs/min-order-v3-%j.out -e logs/min-order-v3-%j.err "
        f"--wrap {json.dumps(wrap)}"
    )


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-root", type=Path, default=default_output_root())
    parser.add_argument("--tasks", type=str, default="add,prod,gap,relmode")
    parser.add_argument("--models", type=str, default="mpnn,mpnn_vn,nope_gt,gt_route,gt_full")
    parser.add_argument("--seeds", type=str, default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--n-nodes", type=int, default=32)
    parser.add_argument("--receptive-radius", type=int, default=4)
    parser.add_argument("--mode-count", type=int, default=6)
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
    parser.add_argument("--methodology-graphs", type=int, default=200)
    parser.add_argument("--k-sweep-values", type=str, default="1,2,3,4,5,6,7,8")
    parser.add_argument("--k-sweep-models", type=str, default="mpnn,mpnn_vn,nope_gt,gt_route,gt_full")
    parser.add_argument("--k-sweep-train-size", type=int, default=3000)
    parser.add_argument("--k-sweep-max-epochs", type=int, default=400)
    parser.add_argument("--k-sweep-patience", type=int, default=80)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--run-channel-attribution", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-methodology", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-k-sweep", action=argparse.BooleanOptionalAction, default=False)
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
        ("run-rq0-demand", "Run the RQ0 oracle demand profile.", run_rq0_demand_profile),
        ("run-source-maps", "Run the RQ1 source-map recovery diagnostic.", run_source_maps),
        ("run-rq4-address-mode", "Run the RQ4 content-addressing diagnostic.", run_rq4_address_mode),
        ("run-k-sweep", "Run the RELMODE K-sweep at fixed head budget.", run_k_sweep),
        ("plot", "Generate all poster figures from cached metrics.", plot_all),
        ("run-all", "Run data, training, clean eval, RQ3, methodology diagnostics, channel attribution, and figures.", run_all),
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
