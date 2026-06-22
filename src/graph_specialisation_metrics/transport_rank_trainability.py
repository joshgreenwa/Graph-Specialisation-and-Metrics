"""Trainability of transport rank in a single graph-transformer layer.

This module implements the experiment described in
``trainability_experiment.md``.  The teacher is an in-class relation operator;
the core comparison is identical full-transport models with learned routing
versus fixed oracle routing.  The runner is intentionally standalone and
resumable for HPC use.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


CONDITIONS = ("full_learned", "full_oracle", "routing_only")
CONDITION_LABELS = {
    "full_learned": "Full-GT, learned routing",
    "full_oracle": "Full-GT, oracle routing",
    "routing_only": "Routing-only, learned",
}
CONDITION_COLORS = {
    "full_learned": "#c2473f",
    "full_oracle": "#3b8b5f",
    "routing_only": "#4f78b5",
}


@dataclass(frozen=True)
class ExperimentConfig:
    output_root: Path = Path("artifacts/transport_rank_trainability")
    n_nodes: int = 256
    heads: int = 4
    content_dim: int = 16
    transport_dim: int = 8
    relation_types: int = 32
    batch_size: int = 128
    eval_batch_size: int = 256
    steps: int = 6000
    smoke_steps: int = 2000
    lr: float = 3.0e-3
    weight_decay: float = 0.0
    seeds: tuple[int, ...] = (0, 1, 2)
    rho_g_sweep: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8)
    smoke_rho_g_sweep: tuple[int, ...] = (1, 4, 8)
    dynamics_rho_g: int = 8
    eval_every: int = 250
    log_every: int = 250
    rank_energy: float = 0.99
    device: str = "cuda"
    dtype: str = "float32"
    compile_model: bool = False
    force: bool = False
    smoke: bool = False


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def parse_int_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in str(text).split(",") if x.strip())


def torch_dtype(name: str) -> torch.dtype:
    if name in {"float32", "fp32"}:
        return torch.float32
    if name in {"float64", "double"}:
        return torch.float64
    if name in {"bfloat16", "bf16"}:
        return torch.bfloat16
    raise ValueError(f"unsupported dtype {name!r}")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")


@dataclass
class Teacher:
    perms: torch.Tensor
    weights: torch.Tensor
    singular_values: torch.Tensor
    ey_floor_h: float
    ey_floor_dr: float
    rank99: int
    rho_g: int


def orthonormal_matrices(count: int, dim: int, generator: torch.Generator) -> torch.Tensor:
    raw = torch.randn(dim * dim, count, generator=generator, dtype=torch.float64)
    q, _ = torch.linalg.qr(raw, mode="reduced")
    return q.T.reshape(count, dim, dim).contiguous().float()


def make_relation_perms(
    *,
    relation_types: int,
    n_nodes: int,
    generator: torch.Generator,
) -> torch.Tensor:
    perms = []
    base = torch.arange(n_nodes, dtype=torch.long)
    for _ in range(int(relation_types)):
        p = torch.randperm(n_nodes, generator=generator)
        # Avoid fixed points where possible so every relation genuinely moves
        # information across node indices.
        fixed = p == base
        if bool(fixed.any()) and n_nodes > 1:
            p = torch.roll(p, shifts=1)
        perms.append(p)
    return torch.stack(perms, dim=0)


def svdvals_relation_stack(weights: torch.Tensor) -> torch.Tensor:
    return torch.linalg.svdvals(weights.reshape(weights.size(0), -1).double()).float()


def rank_from_singular_values(s: torch.Tensor, energy: float = 0.99) -> int:
    vals = (s.detach().cpu().double() ** 2)
    total = float(vals.sum().item())
    if total <= 0:
        return 0
    frac = torch.cumsum(vals, dim=0) / total
    return int(torch.searchsorted(frac, torch.tensor(float(energy), dtype=frac.dtype)).item() + 1)


def ey_floor_from_singular_values(s: torch.Tensor, rank: int) -> float:
    vals = (s.detach().cpu().double() ** 2)
    total = float(vals.sum().item())
    if total <= 0:
        return 0.0
    return float(vals[int(rank) :].sum().item() / total)


def make_teacher(config: ExperimentConfig, *, rho_g: int, seed: int, device: torch.device) -> Teacher:
    if config.relation_types % config.heads != 0:
        raise ValueError("relation_types must be divisible by heads")
    group_size = config.relation_types // config.heads
    if group_size < rho_g:
        raise ValueError("R/H must be >= rho_g for full-column-rank group coefficients")
    gen = torch.Generator(device="cpu").manual_seed(int(seed) + 17_003 + 97 * int(rho_g))
    perms = make_relation_perms(
        relation_types=config.relation_types,
        n_nodes=config.n_nodes,
        generator=gen,
    )
    basis_by_head = []
    coeff_by_group = []
    weights = torch.zeros(config.relation_types, config.content_dim, config.content_dim)
    for head in range(config.heads):
        basis = orthonormal_matrices(int(rho_g), config.content_dim, gen)
        coeff_raw = torch.randn(group_size, int(rho_g), generator=gen, dtype=torch.float64)
        q, _ = torch.linalg.qr(coeff_raw, mode="reduced")
        coeff = q[:, : int(rho_g)].float() * math.sqrt(float(group_size))
        start = head * group_size
        end = start + group_size
        weights[start:end] = torch.einsum("rc,cij->rij", coeff, basis)
        basis_by_head.append(basis)
        coeff_by_group.append(coeff)
    weights = weights / math.sqrt(float(rho_g))
    singular = svdvals_relation_stack(weights)
    return Teacher(
        perms=perms.to(device),
        weights=weights.to(device),
        singular_values=singular,
        ey_floor_h=ey_floor_from_singular_values(singular, config.heads),
        ey_floor_dr=ey_floor_from_singular_values(singular, config.transport_dim),
        rank99=rank_from_singular_values(singular, config.rank_energy),
        rho_g=int(rho_g),
    )


def gather_relation_sources(x: torch.Tensor, perms: torch.Tensor) -> torch.Tensor:
    # x: [B, N, d], perms: [R, N] with perms[r, i] = source node for receiver i.
    return x[:, perms, :]


def teacher_forward(x: torch.Tensor, teacher: Teacher) -> torch.Tensor:
    src = gather_relation_sources(x, teacher.perms)
    return torch.einsum("brni,roi->bno", src, teacher.weights)


class TransportRankLayer(nn.Module):
    def __init__(
        self,
        *,
        condition: str,
        heads: int,
        relation_types: int,
        content_dim: int,
        transport_dim: int,
        oracle_group_size: int,
    ) -> None:
        super().__init__()
        self.condition = str(condition)
        self.heads = int(heads)
        self.relation_types = int(relation_types)
        self.content_dim = int(content_dim)
        self.transport_dim = int(transport_dim)
        self.oracle_group_size = int(oracle_group_size)
        if condition not in CONDITIONS:
            raise ValueError(f"unknown condition {condition!r}")
        self.routing_logits = nn.Parameter(0.01 * torch.randn(heads, relation_types))
        if condition in {"full_learned", "full_oracle"}:
            self.rho = nn.Parameter(0.02 * torch.randn(relation_types, transport_dim))
            self.bases = nn.Parameter(
                torch.randn(heads, transport_dim, content_dim, content_dim)
                / math.sqrt(float(content_dim * transport_dim))
            )
        else:
            self.values = nn.Parameter(
                torch.randn(heads, content_dim, content_dim) / math.sqrt(float(content_dim))
            )

    def routing(self) -> torch.Tensor:
        if self.condition == "full_oracle":
            a = self.routing_logits.new_zeros(self.heads, self.relation_types)
            for head in range(self.heads):
                start = head * self.oracle_group_size
                end = start + self.oracle_group_size
                a[head, start:end] = 1.0 / float(self.oracle_group_size)
            return a
        return torch.softmax(self.routing_logits, dim=-1)

    def relation_operators(self) -> torch.Tensor:
        a = self.routing()
        if self.condition in {"full_learned", "full_oracle"}:
            per_head = torch.einsum("rc,hcoi->hroi", self.rho, self.bases)
            return torch.einsum("hr,hroi->roi", a, per_head)
        return torch.einsum("hr,hoi->roi", a, self.values)

    def forward(self, x: torch.Tensor, perms: torch.Tensor) -> torch.Tensor:
        src = gather_relation_sources(x, perms)
        ops = self.relation_operators()
        return torch.einsum("brni,roi->bno", src, ops)

    def mean_entropy(self) -> float:
        a = self.routing().detach().float().clamp_min(1.0e-12)
        return float((-(a * a.log()).sum(dim=-1)).mean().cpu())

    def routing_grad_norm(self) -> float:
        if self.condition == "full_oracle" or self.routing_logits.grad is None:
            return 0.0
        return float(self.routing_logits.grad.detach().norm().cpu())

    def transport_grad_norm(self) -> float:
        vals = []
        for name, param in self.named_parameters():
            if name == "routing_logits" or param.grad is None:
                continue
            vals.append(param.grad.detach().norm().double() ** 2)
        if not vals:
            return 0.0
        return float(torch.sqrt(torch.stack(vals).sum()).cpu())


@torch.no_grad()
def evaluate_model(
    model: TransportRankLayer,
    teacher: Teacher,
    eval_x: torch.Tensor,
    *,
    rank_energy: float,
) -> dict[str, Any]:
    model.eval()
    pred = model(eval_x, teacher.perms)
    target = teacher_forward(eval_x, teacher)
    rel = float(((pred - target) ** 2).sum().detach().cpu() / (target**2).sum().detach().cpu().clamp_min(1.0e-12))
    ops = model.relation_operators().detach().cpu()
    singular = svdvals_relation_stack(ops)
    rank = rank_from_singular_values(singular, rank_energy)
    return {
        "rel_mse": rel,
        "realised_rank": rank,
        "entropy": model.mean_entropy(),
        "singular_values": [float(x) for x in singular.cpu().tolist()],
    }


def run_key(config: ExperimentConfig, *, condition: str, rho_g: int, seed: int) -> str:
    return f"{condition}_rho{int(rho_g):02d}_seed{int(seed)}"


def run_path(config: ExperimentConfig, *, condition: str, rho_g: int, seed: int) -> Path:
    return ensure_dir(config.output_root / "runs") / f"{run_key(config, condition=condition, rho_g=rho_g, seed=seed)}.json"


def train_one(
    config: ExperimentConfig,
    *,
    condition: str,
    rho_g: int,
    seed: int,
    dynamics: bool = False,
) -> dict[str, Any]:
    path = run_path(config, condition=condition, rho_g=rho_g, seed=seed)
    if path.exists() and not config.force:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    set_seed(int(seed))
    device = torch.device(config.device if config.device != "cuda" or torch.cuda.is_available() else "cpu")
    dtype = torch_dtype(config.dtype)
    if dtype != torch.float32:
        print(f"[warning] dtype={config.dtype} requested; rank metrics are still evaluated in float32.")
    teacher = make_teacher(config, rho_g=int(rho_g), seed=int(seed), device=device)
    group_size = config.relation_types // config.heads
    model = TransportRankLayer(
        condition=condition,
        heads=config.heads,
        relation_types=config.relation_types,
        content_dim=config.content_dim,
        transport_dim=config.transport_dim,
        oracle_group_size=group_size,
    ).to(device)
    if config.compile_model:
        model = torch.compile(model)  # type: ignore[assignment]
    steps = config.smoke_steps if config.smoke else config.steps
    opt = torch.optim.Adam(model.parameters(), lr=float(config.lr), weight_decay=float(config.weight_decay))
    gen = torch.Generator(device=device).manual_seed(int(seed) + 44_001)
    eval_x = torch.randn(config.eval_batch_size, config.n_nodes, config.content_dim, generator=gen, device=device)
    history = []
    final_eval: dict[str, Any] = {}
    for step in range(1, int(steps) + 1):
        model.train()
        x = torch.randn(config.batch_size, config.n_nodes, config.content_dim, generator=gen, device=device)
        y = teacher_forward(x, teacher)
        pred = model(x, teacher.perms)
        loss = F.mse_loss(pred, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        routing_grad = model.routing_grad_norm()
        transport_grad = model.transport_grad_norm()
        opt.step()
        should_log = dynamics or step == 1 or step == steps or step % int(config.eval_every) == 0
        if should_log:
            ev = evaluate_model(model, teacher, eval_x, rank_energy=config.rank_energy)
            final_eval = ev
            history.append(
                {
                    "step": int(step),
                    "loss": float(loss.detach().cpu()),
                    "rel_mse": float(ev["rel_mse"]),
                    "realised_rank": int(ev["realised_rank"]),
                    "entropy": float(ev["entropy"]),
                    "routing_grad_norm": float(routing_grad),
                    "transport_grad_norm": float(transport_grad),
                }
            )
        if step % max(int(config.log_every), 1) == 0 or step == 1 or step == steps:
            rel = final_eval.get("rel_mse", float("nan"))
            rank = final_eval.get("realised_rank", -1)
            print(
                f"[train] {condition} rho_g={rho_g} r*={config.heads * int(rho_g)} "
                f"seed={seed} step={step}/{steps} relMSE={rel:.4g} rank={rank} "
                f"H={model.mean_entropy():.3f}"
            )
    final_eval = evaluate_model(model, teacher, eval_x, rank_energy=config.rank_energy)
    payload = {
        "condition": condition,
        "condition_label": CONDITION_LABELS[condition],
        "seed": int(seed),
        "rho_g": int(rho_g),
        "demand_rank": int(config.heads * int(rho_g)),
        "teacher_rank99": int(teacher.rank99),
        "teacher_ey_floor_h": float(teacher.ey_floor_h),
        "teacher_ey_floor_dr": float(teacher.ey_floor_dr),
        "rel_mse": float(final_eval["rel_mse"]),
        "realised_rank": int(final_eval["realised_rank"]),
        "entropy": float(final_eval["entropy"]),
        "singular_values": final_eval["singular_values"],
        "teacher_singular_values": [float(x) for x in teacher.singular_values.cpu().tolist()],
        "history": history,
        "config": {**asdict(config), "output_root": str(config.output_root)},
    }
    write_json(path, payload)
    checkpoint = ensure_dir(config.output_root / "checkpoints") / f"{run_key(config, condition=condition, rho_g=rho_g, seed=seed)}.pt"
    torch.save({"model_state": model.state_dict(), "payload": payload}, checkpoint)
    return payload


def sweep_rho_g(config: ExperimentConfig) -> tuple[int, ...]:
    return config.smoke_rho_g_sweep if config.smoke else config.rho_g_sweep


def sweep_seeds(config: ExperimentConfig) -> tuple[int, ...]:
    return (config.seeds[0],) if config.smoke else config.seeds


def collect_run_rows(config: ExperimentConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    for path in sorted((config.output_root / "runs").glob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        row = {
            key: payload[key]
            for key in [
                "condition",
                "condition_label",
                "seed",
                "rho_g",
                "demand_rank",
                "teacher_rank99",
                "teacher_ey_floor_h",
                "teacher_ey_floor_dr",
                "rel_mse",
                "realised_rank",
                "entropy",
            ]
        }
        summary_rows.append(row)
        for hist in payload.get("history", []):
            history_rows.append(
                {
                    "condition": payload["condition"],
                    "condition_label": payload["condition_label"],
                    "seed": int(payload["seed"]),
                    "rho_g": int(payload["rho_g"]),
                    "demand_rank": int(payload["demand_rank"]),
                    "teacher_ey_floor_dr": float(payload["teacher_ey_floor_dr"]),
                    **hist,
                }
            )
    return summary_rows, history_rows


def aggregate_rows(rows: Sequence[Mapping[str, Any]], keys: Sequence[str], value: str) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[float]] = {}
    for row in rows:
        group = tuple(row[k] for k in keys)
        groups.setdefault(group, []).append(float(row[value]))
    out = []
    for group, vals in sorted(groups.items()):
        arr = np.asarray(vals, dtype=float)
        rec = {k: v for k, v in zip(keys, group, strict=True)}
        rec.update(
            {
                f"{value}_mean": float(arr.mean()),
                f"{value}_std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
                "n": int(arr.size),
            }
        )
        out.append(rec)
    return out


def import_plotting():
    import matplotlib

    if not os.environ.get("DISPLAY"):
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def plot_fig1(config: ExperimentConfig, rows: Sequence[Mapping[str, Any]]) -> Path:
    plt = import_plotting()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), sharex=True)
    rank_agg = aggregate_rows(rows, ("condition", "condition_label", "demand_rank"), "realised_rank")
    mse_agg = aggregate_rows(rows, ("condition", "condition_label", "demand_rank"), "rel_mse")
    floors = {}
    for row in rows:
        floors[int(row["demand_rank"])] = (
            float(row["teacher_ey_floor_dr"]),
            float(row["teacher_ey_floor_h"]),
        )
    xs_all = sorted(floors)
    for condition in CONDITIONS:
        rsub = [r for r in rank_agg if r["condition"] == condition]
        if not rsub:
            continue
        rsub = sorted(rsub, key=lambda r: int(r["demand_rank"]))
        xs = np.asarray([int(r["demand_rank"]) for r in rsub])
        ys = np.asarray([float(r["realised_rank_mean"]) for r in rsub])
        es = np.asarray([float(r["realised_rank_std"]) for r in rsub])
        axes[0].errorbar(
            xs,
            ys,
            yerr=es,
            marker="o",
            linewidth=2.0,
            capsize=2,
            label=CONDITION_LABELS[condition],
            color=CONDITION_COLORS[condition],
        )
        msub = sorted([r for r in mse_agg if r["condition"] == condition], key=lambda r: int(r["demand_rank"]))
        my = np.asarray([float(r["rel_mse_mean"]) for r in msub])
        me = np.asarray([float(r["rel_mse_std"]) for r in msub])
        axes[1].errorbar(
            xs,
            my,
            yerr=me,
            marker="o",
            linewidth=2.0,
            capsize=2,
            label=CONDITION_LABELS[condition],
            color=CONDITION_COLORS[condition],
        )
    axes[0].plot(xs_all, xs_all, color="#555555", linestyle="--", linewidth=1.2, label="demanded rank")
    axes[0].axhline(config.transport_dim, color="#777777", linestyle=":", linewidth=1.1, label=f"d_r={config.transport_dim}")
    axes[0].axhline(config.heads, color="#999999", linestyle="-.", linewidth=1.1, label=f"H={config.heads}")
    axes[0].set_ylabel("realised transport rank (99% energy)")
    axes[0].set_xlabel("demanded rank r*")
    axes[0].set_title("Achieved Rank")
    ey_dr = [floors[x][0] for x in xs_all]
    ey_h = [floors[x][1] for x in xs_all]
    axes[1].plot(xs_all, ey_dr, color="#777777", linestyle=":", linewidth=1.6, label=f"EY floor rank d_r={config.transport_dim}")
    axes[1].plot(xs_all, ey_h, color="#999999", linestyle="-.", linewidth=1.4, label=f"EY floor rank H={config.heads}")
    axes[1].set_ylabel("held-out relMSE")
    axes[1].set_xlabel("demanded rank r*")
    axes[1].set_title("Prediction Error")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    axes[1].legend(frameon=False, fontsize=7, loc="upper left")
    fig.suptitle("Trainability Gap in Relation-Conditioned Transport Rank", y=1.02)
    fig.tight_layout()
    path = ensure_dir(config.output_root / "figures") / "fig1_rank_and_error_vs_demand.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {path}")
    return path


def plot_fig2(config: ExperimentConfig, history: Sequence[Mapping[str, Any]]) -> Path:
    plt = import_plotting()
    demand = config.heads * config.dynamics_rho_g
    rows = [
        r
        for r in history
        if int(r["demand_rank"]) == int(demand) and r["condition"] in {"full_learned", "full_oracle"}
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.7))
    for condition in ("full_learned", "full_oracle"):
        sub = [r for r in rows if r["condition"] == condition]
        if not sub:
            continue
        by_step: dict[int, list[Mapping[str, Any]]] = {}
        for row in sub:
            by_step.setdefault(int(row["step"]), []).append(row)
        steps = sorted(by_step)
        rel = np.asarray([np.mean([float(x["rel_mse"]) for x in by_step[s]]) for s in steps])
        ent = np.asarray([np.mean([float(x["entropy"]) for x in by_step[s]]) for s in steps])
        axes[0].plot(steps, rel, color=CONDITION_COLORS[condition], linewidth=2.0, label=CONDITION_LABELS[condition])
        axes[1].plot(steps, ent, color=CONDITION_COLORS[condition], linewidth=2.0, label=CONDITION_LABELS[condition])
    ey = float(rows[0]["teacher_ey_floor_dr"]) if rows else float("nan")
    axes[0].axhline(ey, color="#777777", linestyle=":", linewidth=1.4, label=f"EY floor rank d_r={config.transport_dim}")
    axes[1].axhline(math.log(config.relation_types), color="#777777", linestyle=":", linewidth=1.3, label="uniform entropy log R")
    axes[1].axhline(math.log(config.relation_types // config.heads), color="#999999", linestyle="-.", linewidth=1.3, label="oracle group entropy")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("training step")
    axes[1].set_xlabel("training step")
    axes[0].set_ylabel("held-out relMSE")
    axes[1].set_ylabel("mean attention entropy")
    axes[0].set_title(f"Optimisation at r*={demand}")
    axes[1].set_title("Routing Specialisation")
    for ax in axes:
        ax.grid(axis="y", color="#dddddd", linewidth=0.6)
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    path = ensure_dir(config.output_root / "figures") / "fig2_training_dynamics.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {path}")
    return path


def plot_fig3(config: ExperimentConfig, history: Sequence[Mapping[str, Any]]) -> Path:
    plt = import_plotting()
    demand = config.heads * config.dynamics_rho_g
    rows = [r for r in history if int(r["demand_rank"]) == int(demand) and r["condition"] == "full_learned"]
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    by_step: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_step.setdefault(int(row["step"]), []).append(row)
    steps = sorted(by_step)
    if steps:
        routing = np.asarray([np.mean([float(x["routing_grad_norm"]) for x in by_step[s]]) for s in steps])
        transport = np.asarray([np.mean([float(x["transport_grad_norm"]) for x in by_step[s]]) for s in steps])
        ax.plot(steps, routing, color="#c2473f", linewidth=2.0, label="routing logits grad")
        ax.plot(steps, transport, color="#4f78b5", linewidth=2.0, label="transport params grad")
    ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("gradient norm")
    ax.set_title(f"Routing-Transport Gradient Starvation at r*={demand}")
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = ensure_dir(config.output_root / "figures") / "fig3_gradient_starvation.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {path}")
    return path


def write_tables(config: ExperimentConfig, rows: Sequence[Mapping[str, Any]], history: Sequence[Mapping[str, Any]]) -> None:
    metrics = ensure_dir(config.output_root / "metrics")
    write_csv(metrics / "trainability_summary.csv", rows)
    write_csv(metrics / "trainability_history.csv", history)
    print("\n[summary] final rank/relMSE")
    for row in sorted(rows, key=lambda r: (int(r["demand_rank"]), str(r["condition"]), int(r["seed"]))):
        print(
            f"r*={int(row['demand_rank']):02d} seed={int(row['seed'])} "
            f"{row['condition']:<13} rank={int(row['realised_rank']):02d} "
            f"relMSE={float(row['rel_mse']):.5g} "
            f"EY(d_r)={float(row['teacher_ey_floor_dr']):.5g}"
        )


def plot_all(config: ExperimentConfig) -> None:
    rows, history = collect_run_rows(config)
    if not rows:
        raise FileNotFoundError(f"No run JSON files found under {config.output_root / 'runs'}")
    write_tables(config, rows, history)
    plot_fig1(config, rows)
    plot_fig2(config, history)
    plot_fig3(config, history)


def run_experiment(config: ExperimentConfig) -> None:
    ensure_dir(config.output_root)
    write_json(config.output_root / "config.json", {**asdict(config), "output_root": str(config.output_root)})
    rho_values = sweep_rho_g(config)
    seeds = sweep_seeds(config)
    for rho_g in rho_values:
        for seed in seeds:
            for condition in CONDITIONS:
                train_one(config, condition=condition, rho_g=int(rho_g), seed=int(seed), dynamics=False)
    # Ensure high-resolution dynamics for the hardest demand.
    dynamics_rho = int(config.dynamics_rho_g)
    for seed in seeds:
        for condition in ("full_learned", "full_oracle"):
            train_one(config, condition=condition, rho_g=dynamics_rho, seed=int(seed), dynamics=True)
    plot_all(config)


def print_hpc_commands(config: ExperimentConfig) -> None:
    root = str(config.output_root)
    print(
        f"""mkdir -p logs

TRAINABILITY_JOB=$(sbatch --parsable \\
  -A mlmi-jgg45-sl2-gpu -p ampere --qos=gpu1 \\
  --gres=gpu:1 --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=80G --time=02:00:00 \\
  -J trainability \\
  -o logs/trainability-%j.out -e logs/trainability-%j.err \\
  --wrap 'source /usr/local/Cluster-Apps/miniconda3/4.5.1/etc/profile.d/conda.sh; conda activate graphbench-algoreas; cd /rds/user/jgg45/hpc-work/Graph-Specialisation-and-Metrics; export PYTHONPATH=$PWD/src:$PYTHONPATH; export OMP_NUM_THREADS=8; python -u -m graph_specialisation_metrics.transport_rank_trainability run --output-root {root} --device cuda')

echo "TRAINABILITY_JOB=$TRAINABILITY_JOB"
sacct -j "$TRAINABILITY_JOB" --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS -X
tail -f logs/trainability-${{TRAINABILITY_JOB}}.out
"""
    )


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        output_root=Path(args.output_root),
        n_nodes=int(args.n_nodes),
        heads=int(args.heads),
        content_dim=int(args.content_dim),
        transport_dim=int(args.transport_dim),
        relation_types=int(args.relation_types),
        batch_size=int(args.batch_size),
        eval_batch_size=int(args.eval_batch_size),
        steps=int(args.steps),
        smoke_steps=int(args.smoke_steps),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        seeds=parse_int_tuple(args.seeds),
        rho_g_sweep=parse_int_tuple(args.rho_g_sweep),
        smoke_rho_g_sweep=parse_int_tuple(args.smoke_rho_g_sweep),
        dynamics_rho_g=int(args.dynamics_rho_g),
        eval_every=int(args.eval_every),
        log_every=int(args.log_every),
        rank_energy=float(args.rank_energy),
        device=str(args.device),
        dtype=str(args.dtype),
        compile_model=bool(args.compile_model),
        force=bool(args.force),
        smoke=bool(args.smoke),
    )


def add_common_args(parser: argparse.ArgumentParser) -> None:
    defaults = ExperimentConfig()
    parser.add_argument("--output-root", type=str, default=str(defaults.output_root))
    parser.add_argument("--n-nodes", type=int, default=defaults.n_nodes)
    parser.add_argument("--heads", type=int, default=defaults.heads)
    parser.add_argument("--content-dim", type=int, default=defaults.content_dim)
    parser.add_argument("--transport-dim", type=int, default=defaults.transport_dim)
    parser.add_argument("--relation-types", type=int, default=defaults.relation_types)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--eval-batch-size", type=int, default=defaults.eval_batch_size)
    parser.add_argument("--steps", type=int, default=defaults.steps)
    parser.add_argument("--smoke-steps", type=int, default=defaults.smoke_steps)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--seeds", type=str, default=",".join(map(str, defaults.seeds)))
    parser.add_argument("--rho-g-sweep", type=str, default=",".join(map(str, defaults.rho_g_sweep)))
    parser.add_argument("--smoke-rho-g-sweep", type=str, default=",".join(map(str, defaults.smoke_rho_g_sweep)))
    parser.add_argument("--dynamics-rho-g", type=int, default=defaults.dynamics_rho_g)
    parser.add_argument("--eval-every", type=int, default=defaults.eval_every)
    parser.add_argument("--log-every", type=int, default=defaults.log_every)
    parser.add_argument("--rank-energy", type=float, default=defaults.rank_energy)
    parser.add_argument("--device", type=str, default=defaults.device)
    parser.add_argument("--dtype", type=str, default=defaults.dtype)
    parser.add_argument("--compile-model", action=argparse.BooleanOptionalAction, default=defaults.compile_model)
    parser.add_argument("--force", action=argparse.BooleanOptionalAction, default=defaults.force)
    parser.add_argument("--smoke", action=argparse.BooleanOptionalAction, default=defaults.smoke)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in [
        ("run", "Run sweep, dynamics, tables, and figures."),
        ("plot", "Regenerate tables and figures from cached run JSON files."),
        ("print-hpc-commands", "Print an A100 Slurm command."),
    ]:
        p = sub.add_parser(name, help=help_text)
        add_common_args(p)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = build_config(args)
    if args.command == "run":
        run_experiment(config)
    elif args.command == "plot":
        plot_all(config)
    elif args.command == "print-hpc-commands":
        print_hpc_commands(config)
    else:
        raise ValueError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
