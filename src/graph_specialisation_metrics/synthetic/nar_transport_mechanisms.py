"""Resumable NAR attention-faithfulness and transport-mechanism analysis.

This module consumes checkpoints produced by :mod:`nar_grit_fixed` and never
modifies their training configuration or fingerprint.  It adds task-specific
payload/address interventions, an exact routing/message decomposition at
official GRIT's routed ``wV`` site, contextual semantic/structural scores,
selected-family ablations, finite hybrid patching, and cached paper figures.

All expensive model results are written as per-cell ``.pt`` files.  Tables and
figures are derived only from those caches, so visual changes never require a
model forward.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from graph_specialisation_metrics.synthetic import nar_grit_fixed as nar


ANALYSIS_VERSION = "nar-transport-mechanisms-v1"
MODEL_ORDER = ("1hop", "2hop", "dense")
MODEL_COLOURS = {"1hop": "#6550a4", "2hop": "#2b8cbe", "dense": "#d7301f"}
MODEL_MARKERS = {"1hop": "o", "2hop": "s", "dense": "D"}
LAYER_COLOURS = {0: "#238b45", 1: "#cb181d"}
N_MARKERS = {4: "o", 8: "s", 16: "^", 32: "P", 64: "D"}
EPS = 1.0e-12

PRIMARY_INTERVENTIONS = ("target_payload", "address_different_answer")
CONTROL_INTERVENTIONS = ("distractor_payload", "address_same_answer")
STRUCTURAL_INTERVENTION = "structural_query_record"


@dataclass(frozen=True)
class AnalysisConfig:
    drive_root: str = nar.DEFAULT_DRIVE_ROOT
    run_name: str = "nar_grit_fixed_n_v3"
    analysis_width: int = 64
    models: tuple[str, ...] = MODEL_ORDER
    ns: tuple[int, ...] = (4, 8, 16, 32, 64)
    seeds: tuple[int, ...] = (0, 1, 2)
    anchor_ns: tuple[int, ...] = (4, 16, 64)
    donors: int = 4
    discovery_graphs: int = 32
    mechanism_graphs: int = 96
    robustness_graphs: int = 48
    causal_graphs: int = 256
    causal_donors: int = 4
    family_size: int = 2
    random_families: int = 8
    max_batch_nodes: int = 4500
    max_dense_pairs: int = 300_000
    max_replica_pairs: int = 1_200_000
    bootstrap_samples: int = 2000
    device: str = "cuda"
    grit_dir: str = nar.DEFAULT_GRIT_DIR
    analysis_seed: int = 2_026_0721
    solved_accuracy: float = 0.85
    closure_atol: float = 2.0e-5
    closure_rtol: float = 2.0e-4
    noop_tol: float = 2.0e-5

    def validate(self) -> None:
        if self.analysis_width not in (64, 128):
            raise ValueError("analysis_width must be 64 or 128")
        if not self.models or any(model not in MODEL_ORDER for model in self.models):
            raise ValueError(f"models must be drawn from {MODEL_ORDER}")
        if not self.ns or any(value <= 1 for value in self.ns):
            raise ValueError("N values must be greater than one")
        if any(value not in self.ns for value in self.anchor_ns):
            raise ValueError("anchor_ns must be a subset of ns")
        if self.donors <= 0 or self.causal_donors <= 0:
            raise ValueError("donor counts must be positive")
        if self.family_size <= 0:
            raise ValueError("family_size must be positive")


@dataclass
class Replica:
    intervention: str
    donor: int
    batch: nar.NarBatch
    valid: torch.Tensor
    clean_label: torch.Tensor
    variant_label: torch.Tensor
    clean_target_idx: torch.Tensor
    variant_target_idx: torch.Tensor


@dataclass
class ReplicaBundle:
    clean: nar.NarBatch
    replicas: list[Replica]

    def combined(self) -> nar.NarBatch:
        return nar.concat_batches([self.clean] + [item.batch for item in self.replicas])

    def indices(self, intervention: str) -> list[int]:
        # Graph-block index 0 is clean; replicas start at 1.
        return [
            index + 1
            for index, item in enumerate(self.replicas)
            if item.intervention == intervention
        ]

    def items(self, intervention: str) -> list[tuple[int, Replica]]:
        return [
            (index + 1, item)
            for index, item in enumerate(self.replicas)
            if item.intervention == intervention
        ]


def parse_int_tuple(value: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        return tuple(int(item.strip()) for item in value.split(",") if item.strip())
    return tuple(map(int, value))


def parse_str_tuple(value: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    return tuple(map(str, value))


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    raise TypeError(f"cannot JSON-encode {type(value)!r}")


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if not rows:
        temporary.write_text("", encoding="utf-8")
        os.replace(temporary, path)
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(str(key))
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def scientific_fingerprint(cfg: AnalysisConfig) -> str:
    payload = asdict(cfg)
    for key in ("drive_root", "run_name", "grit_dir", "bootstrap_samples"):
        payload.pop(key, None)
    encoded = json.dumps(
        {"version": ANALYSIS_VERSION, **payload}, sort_keys=True, default=str
    ).encode()
    return hashlib.sha1(encoded).hexdigest()[:16]


def analysis_root(cfg: AnalysisConfig) -> Path:
    return (
        Path(cfg.drive_root)
        / cfg.run_name
        / "transport_mechanisms_v1"
        / f"d{cfg.analysis_width}"
    )


def run_root(cfg: AnalysisConfig) -> Path:
    return Path(cfg.drive_root) / cfg.run_name


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def resolve_device(requested: str) -> torch.device:
    if str(requested).startswith("cuda") and not torch.cuda.is_available():
        print("[device] CUDA unavailable; falling back to CPU", flush=True)
        return torch.device("cpu")
    return torch.device(requested)


def base_nar_config(payload: Mapping[str, Any]) -> nar.Config:
    saved = dict(payload.get("config", {}))
    allowed = {item.name for item in fields(nar.Config)}
    values = {key: value for key, value in saved.items() if key in allowed}
    return nar.Config(**values)


def checkpoint_metadata(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = ("model_name", "width", "N", "seed", "state_dict", "best_validation", "heldout")
    missing = [key for key in required if key not in payload]
    if missing:
        raise RuntimeError(f"checkpoint {path} is missing {missing}")
    validation = dict(payload["best_validation"])
    heldout = dict(payload["heldout"])
    return {
        "path": str(path),
        "experiment_version": str(payload.get("version", "")),
        "fingerprint": str(payload.get("fingerprint", "")),
        "official_grit_commit": str(payload.get("official_grit_commit", "")),
        "model": str(payload["model_name"]),
        "width": int(payload["width"]),
        "N": int(payload["N"]),
        "seed": int(payload["seed"]),
        "validation_loss": float(validation.get("loss", float("nan"))),
        "validation_accuracy": float(validation.get("accuracy", float("nan"))),
        "heldout_loss": float(heldout.get("loss", float("nan"))),
        "heldout_accuracy": float(heldout.get("accuracy", float("nan"))),
        "parameters": int(payload.get("parameters", 0)),
        "mtime": float(path.stat().st_mtime),
    }


def build_checkpoint_manifest(cfg: AnalysisConfig, *, force: bool = False) -> list[dict[str, Any]]:
    root = analysis_root(cfg)
    manifest_path = root / "checkpoint_manifest.csv"
    json_path = root / "checkpoint_manifest.json"
    if json_path.exists() and not force:
        cached = json.loads(json_path.read_text(encoding="utf-8"))
        if cached.get("analysis_version") == ANALYSIS_VERSION:
            rows = list(cached.get("rows", []))
            if rows and all(Path(row["path"]).exists() for row in rows):
                return rows

    candidates = sorted((run_root(cfg) / "checkpoints").glob("*.pt"))
    if not candidates:
        raise FileNotFoundError(f"no checkpoints found under {run_root(cfg) / 'checkpoints'}")
    rows = []
    for path in candidates:
        row = checkpoint_metadata(path)
        if (
            row["width"] == cfg.analysis_width
            and row["model"] in cfg.models
            and row["N"] in cfg.ns
            and row["seed"] in cfg.seeds
        ):
            rows.append(row)

    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["N"], row["seed"])].append(row)
    expected = {
        (model, records, seed)
        for model in cfg.models
        for records in cfg.ns
        for seed in cfg.seeds
    }
    missing = sorted(expected - set(grouped))
    duplicates = {key: value for key, value in grouped.items() if len(value) != 1}
    if missing:
        raise FileNotFoundError(f"missing width-{cfg.analysis_width} checkpoints: {missing}")
    if duplicates:
        detail = {key: [row["path"] for row in value] for key, value in duplicates.items()}
        raise RuntimeError(f"ambiguous checkpoint cells: {detail}")

    by_cell: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cell[(row["model"], row["N"])].append(row)
    for cell_rows in by_cell.values():
        if not all(math.isfinite(float(row["validation_loss"])) for row in cell_rows):
            raise RuntimeError("validation loss is required for checkpoint selection")
        best = min(cell_rows, key=lambda item: (item["validation_loss"], item["seed"]))
        for row in cell_rows:
            row["selected_for_analysis"] = row is best

    for records in cfg.ns:
        counts = {
            int(row["parameters"])
            for row in rows
            if int(row["N"]) == records
        }
        if len(counts) != 1:
            raise RuntimeError(
                f"support variants are not parameter matched at width={cfg.analysis_width}, "
                f"N={records}: {sorted(counts)}"
            )

    rows.sort(key=lambda item: (MODEL_ORDER.index(item["model"]), item["N"], item["seed"]))
    write_csv(manifest_path, rows)
    atomic_write_json(
        json_path,
        {
            "analysis_version": ANALYSIS_VERSION,
            "width": cfg.analysis_width,
            "created": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "rows": rows,
        },
    )
    return rows


def selected_manifest_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows if bool(row["selected_for_analysis"])]


def analysis_schedule(
    manifest: Sequence[Mapping[str, Any]], cfg: AnalysisConfig
) -> list[dict[str, Any]]:
    scheduled: dict[tuple[str, int, int], dict[str, Any]] = {}
    for row in manifest:
        key = (str(row["model"]), int(row["N"]), int(row["seed"]))
        if bool(row["selected_for_analysis"]):
            scheduled[key] = {
                **dict(row),
                "sample_type": "primary",
                "graphs": cfg.mechanism_graphs,
            }
        elif int(row["N"]) in cfg.anchor_ns:
            scheduled[key] = {
                **dict(row),
                "sample_type": "robustness",
                "graphs": cfg.robustness_graphs,
            }
    return sorted(
        scheduled.values(),
        key=lambda item: (MODEL_ORDER.index(item["model"]), item["N"], item["seed"]),
    )


def load_model_from_manifest(
    row: Mapping[str, Any], device: torch.device
) -> tuple[Any, dict[str, Any], nar.Config]:
    payload = torch.load(Path(row["path"]), map_location="cpu", weights_only=False)
    config = base_nar_config(payload)
    model_class = nar.build_model_class()
    model = model_class(
        config,
        str(payload["model_name"]),
        int(payload["width"]),
        int(payload["N"]),
    ).to(device)
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint state mismatch for {row['path']}: missing={missing}, unexpected={unexpected}"
        )
    model.eval()
    return model, payload, config


def _base_replica(intervention: str, donor: int, batch: nar.NarBatch) -> Replica:
    clean_label = batch.y.clone()
    target = batch.target_idx.clone()
    return Replica(
        intervention=intervention,
        donor=int(donor),
        batch=nar.clone_batch(batch),
        valid=torch.ones(len(batch), dtype=torch.bool),
        clean_label=clean_label,
        variant_label=clean_label.clone(),
        clean_target_idx=target,
        variant_target_idx=target.clone(),
    )


def target_payload_replica(batch: nar.NarBatch, records: int, donor: int) -> Replica:
    item = _base_replica("target_payload", donor, batch)
    for graph in range(len(batch)):
        node = int(batch.target_idx[graph])
        original = int(batch.x[graph, node, 1])
        offset = 1 + (int(donor) % max(1, int(records) - 1))
        item.batch.x[graph, node, 1] = (original + offset) % int(records)
    return item


def _address_candidates(
    batch: nar.NarBatch,
    graph: int,
    *,
    same_answer: bool,
) -> list[int]:
    target = int(batch.target_idx[graph])
    answer = int(batch.y[graph])
    out = []
    for node in torch.where(batch.record_mask[graph])[0].tolist():
        if int(node) == target:
            continue
        value = int(batch.x[graph, int(node), 1])
        if (value == answer) == bool(same_answer):
            out.append(int(node))
    return out


def address_replica(
    batch: nar.NarBatch,
    records: int,
    donor: int,
    *,
    same_answer: bool,
) -> Replica:
    del records
    name = "address_same_answer" if same_answer else "address_different_answer"
    item = _base_replica(name, donor, batch)
    for graph in range(len(batch)):
        candidates = _address_candidates(batch, graph, same_answer=same_answer)
        if not candidates:
            item.valid[graph] = False
            continue
        node = candidates[int(donor) % len(candidates)]
        query = int(batch.query_idx[graph])
        item.batch.x[graph, query, 0] = batch.x[graph, node, 0]
        item.batch.target_idx[graph] = node
        item.batch.y[graph] = batch.x[graph, node, 1]
        item.variant_target_idx[graph] = node
        item.variant_label[graph] = batch.x[graph, node, 1]
    return item


def distractor_payload_replica(
    batch: nar.NarBatch, records: int, donor: int
) -> Replica:
    item = _base_replica("distractor_payload", donor, batch)
    for graph in range(len(batch)):
        target = int(batch.target_idx[graph])
        candidates = [
            int(node)
            for node in torch.where(batch.record_mask[graph])[0].tolist()
            if int(node) != target
        ]
        node = candidates[int(donor) % len(candidates)]
        original = int(batch.x[graph, node, 1])
        offset = 1 + (int(donor) % max(1, int(records) - 1))
        item.batch.x[graph, node, 1] = (original + offset) % int(records)
    return item


def _swap_rrwp_role(rrwp: torch.Tensor, left: int, right: int) -> torch.Tensor:
    order = torch.arange(rrwp.size(0), device=rrwp.device)
    order[left], order[right] = order[right].clone(), order[left].clone()
    return rrwp.index_select(0, order).index_select(1, order)


def structural_query_record_replica(
    batch: nar.NarBatch, donor: int
) -> Replica:
    item = _base_replica(STRUCTURAL_INTERVENTION, donor, batch)
    for graph in range(len(batch)):
        query = int(batch.query_idx[graph])
        records = torch.where(batch.record_mask[graph])[0].tolist()
        partner = int(records[int(donor) % len(records)])
        item.batch.rrwp[graph] = _swap_rrwp_role(batch.rrwp[graph], query, partner)
    return item


def structural_record_record_noop(batch: nar.NarBatch, donor: int) -> Replica:
    item = _base_replica("structural_record_record_noop", donor, batch)
    for graph in range(len(batch)):
        records = torch.where(batch.record_mask[graph])[0].tolist()
        left = int(records[int(donor) % len(records)])
        right = int(records[(int(donor) + 1) % len(records)])
        item.batch.rrwp[graph] = _swap_rrwp_role(batch.rrwp[graph], left, right)
    return item


def identical_replica(batch: nar.NarBatch) -> Replica:
    return _base_replica("identical", 0, batch)


def record_permutation_replica(batch: nar.NarBatch) -> Replica:
    item = _base_replica("record_permutation", 0, batch)
    for graph in range(len(batch)):
        nodes = torch.where(batch.record_mask[graph])[0]
        item.batch.x[graph, nodes] = torch.roll(batch.x[graph, nodes], shifts=1, dims=0)
        query_key = int(batch.x[graph, int(batch.query_idx[graph]), 0])
        keys = item.batch.x[graph, nodes, 0]
        match = torch.where(keys == query_key)[0]
        if int(match.numel()) != 1:
            raise RuntimeError("record permutation lost the unique queried key")
        item.batch.target_idx[graph] = nodes[int(match[0])]
        item.variant_target_idx[graph] = item.batch.target_idx[graph]
    return item


def build_replica_bundle(
    batch: nar.NarBatch,
    records: int,
    donors: int,
    *,
    include_controls: bool = True,
    include_structure: bool = True,
) -> ReplicaBundle:
    replicas: list[Replica] = []
    for donor in range(int(donors)):
        replicas.append(target_payload_replica(batch, records, donor))
        replicas.append(address_replica(batch, records, donor, same_answer=False))
        if include_controls:
            replicas.append(distractor_payload_replica(batch, records, donor))
            replicas.append(address_replica(batch, records, donor, same_answer=True))
        if include_structure:
            replicas.append(structural_query_record_replica(batch, donor))
    replicas.append(identical_replica(batch))
    if include_controls:
        replicas.append(record_permutation_replica(batch))
        replicas.append(structural_record_record_noop(batch, 0))
    return ReplicaBundle(clean=nar.clone_batch(batch), replicas=replicas)


def verify_replica(clean: nar.NarBatch, replica: Replica) -> dict[str, float]:
    name = replica.intervention
    adj_max = float((replica.batch.adj - clean.adj).abs().max())
    rrwp_max = float((replica.batch.rrwp - clean.rrwp).abs().max())
    changed_x = (replica.batch.x != clean.x)
    changed_keys = int(changed_x[..., 0].sum())
    changed_values = int(changed_x[..., 1].sum())
    if adj_max != 0.0:
        raise RuntimeError(f"{name} changed adjacency")
    if name.startswith("structural_"):
        if changed_keys or changed_values:
            raise RuntimeError(f"{name} changed semantic content")
    else:
        if rrwp_max != 0.0:
            raise RuntimeError(f"{name} changed RRWP")
    if name == "target_payload" and changed_values != int(replica.valid.sum()):
        raise RuntimeError("target payload did not change exactly one value per valid graph")
    if name.startswith("address_") and changed_values:
        raise RuntimeError(f"{name} changed a memory value")
    if name == "distractor_payload" and changed_values != int(replica.valid.sum()):
        raise RuntimeError("distractor payload did not change exactly one value per graph")
    if name == "identical" and (changed_keys or changed_values or rrwp_max):
        raise RuntimeError("identical replica is not identical")
    return {
        "adj_max": adj_max,
        "rrwp_max": rrwp_max,
        "changed_keys": float(changed_keys),
        "changed_values": float(changed_values),
        "valid_graphs": float(replica.valid.sum()),
    }


def symmetric_output_decomposition(
    attention_clean: torch.Tensor,
    message_clean: torch.Tensor,
    attention_variant: torch.Tensor,
    message_variant: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return exact clean-minus-variant routing, message and total head outputs.

    Expected shapes are ``A=[B,H,N,N]`` and ``m=[B,H,N,N,D]``.  Returned
    tensors have shape ``[B,H,N,D]``.
    """

    a_bar = 0.5 * (attention_clean + attention_variant)
    m_bar = 0.5 * (message_clean + message_variant)
    route = ((attention_clean - attention_variant).unsqueeze(-1) * m_bar).sum(dim=3)
    message = (a_bar.unsqueeze(-1) * (message_clean - message_variant)).sum(dim=3)
    clean_output = (attention_clean.unsqueeze(-1) * message_clean).sum(dim=3)
    variant_output = (attention_variant.unsqueeze(-1) * message_variant).sum(dim=3)
    total = clean_output - variant_output
    return route, message, total


def project_components(
    phi: torch.Tensor,
    delta_route: torch.Tensor,
    delta_message: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Project component deltas with the production per-node Method-A estimator.

    ``phi`` has shape ``[T,B,N,H,D]`` and deltas are ``[B,H,N,D]``.
    Returned scores have shape ``[B,H]``.
    """

    route_bnhd = delta_route.permute(0, 2, 1, 3)
    message_bnhd = delta_message.permute(0, 2, 1, 3)
    projected_route = torch.einsum("tbnhd,bnhd->tbnh", phi, route_bnhd)
    projected_message = torch.einsum("tbnhd,bnhd->tbnh", phi, message_bnhd)
    route_score = torch.linalg.vector_norm(projected_route, dim=0).sum(dim=1)
    message_score = torch.linalg.vector_norm(projected_message, dim=0).sum(dim=1)
    total_score = torch.linalg.vector_norm(
        projected_route + projected_message, dim=0
    ).sum(dim=1)
    denominator = (route_score + message_score).clamp_min(EPS)
    return {
        "route": route_score,
        "message": message_score,
        "total": total_score,
        "routing_share": route_score / denominator,
        "mechanism_balance": (route_score - message_score) / denominator,
        "alignment": total_score / denominator,
        "projected_route": projected_route,
        "projected_message": projected_message,
    }


def batched_output_jacobian(
    logits: torch.Tensor,
    routed: torch.Tensor,
    *,
    clean_graphs: int,
    nodes: int,
    retain_graph: bool,
) -> torch.Tensor:
    """Return ``d logits_t / d wV`` as ``[T,B,N,H,D]``.

    A batched VJP avoids one Python/autograd traversal per output class.  A
    compatibility fallback is retained for older torch builds.
    """

    outputs = logits[:clean_graphs].sum(dim=0)
    targets = torch.eye(outputs.numel(), device=outputs.device, dtype=outputs.dtype)
    try:
        gradient = torch.autograd.grad(
            outputs,
            routed,
            grad_outputs=targets,
            retain_graph=retain_graph,
            is_grads_batched=True,
        )[0]
    except (TypeError, RuntimeError):
        pieces = []
        for target in range(int(outputs.numel())):
            pieces.append(
                torch.autograd.grad(
                    outputs[target],
                    routed,
                    retain_graph=True if target + 1 < outputs.numel() else retain_graph,
                )[0]
            )
        gradient = torch.stack(pieces, dim=0)
    clean_nodes = int(clean_graphs) * int(nodes)
    return gradient[:, :clean_nodes].reshape(
        outputs.numel(), clean_graphs, nodes, routed.size(-2), routed.size(-1)
    )


def finite_mean(values: torch.Tensor, valid: torch.Tensor, dim: int = 0) -> torch.Tensor:
    weights = valid.to(device=values.device, dtype=values.dtype)
    while weights.dim() < values.dim():
        weights = weights.unsqueeze(-1)
    return (values * weights).sum(dim=dim) / weights.sum(dim=dim).clamp_min(1.0)


def mean_ci(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return float("nan"), float("nan")
    if len(array) == 1:
        return float(array[0]), 0.0
    t95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(len(array), 1.96)
    return float(array.mean()), float(t95 * array.std(ddof=1) / math.sqrt(len(array)))


def rankdata(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=float)
    start = 0
    while start < len(array):
        stop = start + 1
        while stop < len(array) and array[order[stop]] == array[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    left, right = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = np.isfinite(left) & np.isfinite(right)
    if int(keep.sum()) < 3:
        return float("nan")
    left_rank, right_rank = rankdata(left[keep]), rankdata(right[keep])
    if np.std(left_rank) == 0 or np.std(right_rank) == 0:
        return float("nan")
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


@dataclass
class DenseLayerCapture:
    layer: int
    attention: torch.Tensor
    message: torch.Tensor
    mask: torch.Tensor
    routed: torch.Tensor


@dataclass
class ForwardCapture:
    logits: torch.Tensor
    layers: list[DenseLayerCapture]


def _densify_sparse_record(
    record: Any,
    *,
    graph_blocks: int,
    graphs_per_block: int,
    nodes: int,
) -> DenseLayerCapture:
    total_graphs = int(graph_blocks) * int(graphs_per_block)
    heads = int(record.attention.size(1))
    dim = int(record.message.size(-1))
    device = record.attention.device
    attention = torch.zeros(
        total_graphs, heads, nodes, nodes, device=device, dtype=torch.float32
    )
    message = torch.zeros(
        total_graphs, heads, nodes, nodes, dim, device=device, dtype=torch.float32
    )
    mask = torch.zeros(
        total_graphs, heads, nodes, nodes, device=device, dtype=torch.bool
    )
    graph = record.graph.long()
    destination = record.local_dst.long()
    source = record.local_src.long()
    attention[graph, :, destination, source] = record.attention.float()
    message[graph, :, destination, source] = record.message.float()
    mask[graph, :, destination, source] = True
    routed = record.head_output.reshape(total_graphs, nodes, heads, dim)
    reconstructed = (attention.unsqueeze(-1) * message).sum(dim=3).permute(0, 2, 1, 3)
    error = float((reconstructed - routed.float()).abs().max().detach().cpu())
    if error > 5.0e-5:
        raise RuntimeError(
            f"captured A/m fields do not reconstruct routed wV at layer {record.layer}: "
            f"max error {error:.3e}"
        )
    return DenseLayerCapture(
        layer=int(record.layer),
        attention=attention.reshape(
            graph_blocks, graphs_per_block, heads, nodes, nodes
        ),
        message=message.reshape(
            graph_blocks, graphs_per_block, heads, nodes, nodes, dim
        ),
        mask=mask.reshape(graph_blocks, graphs_per_block, heads, nodes, nodes),
        routed=record.head_output,
    )


def capture_analysis_forward(
    model: Any,
    combined: nar.NarBatch,
    *,
    graph_blocks: int,
    graphs_per_block: int,
    nodes: int,
    device: torch.device,
) -> ForwardCapture:
    from graph_specialisation_metrics.mechanistic_operator_analysis import (
        OfficialGRITMechanisticCollector,
    )

    model.eval()
    moved = combined.to(device)
    with OfficialGRITMechanisticCollector(model) as collector:
        logits = model(moved)
    records = sorted(collector.records, key=lambda item: item.layer)
    if len(records) != int(model.L):
        raise RuntimeError(f"expected {model.L} GRIT layers, captured {len(records)}")
    layers = [
        _densify_sparse_record(
            record,
            graph_blocks=graph_blocks,
            graphs_per_block=graphs_per_block,
            nodes=nodes,
        )
        for record in records
    ]
    return ForwardCapture(
        logits=logits.reshape(graph_blocks, graphs_per_block, -1),
        layers=layers,
    )


def clean_attention_selection(
    attention: torch.Tensor,
    batch: nar.NarBatch,
) -> dict[str, torch.Tensor]:
    """Clean target-vs-background attention for ``A=[B,H,N,N]``."""

    graphs, heads = int(attention.size(0)), int(attention.size(1))
    advantage = torch.full((graphs, heads), float("nan"), device=attention.device)
    ratio = torch.full_like(advantage, float("nan"))
    for graph in range(graphs):
        centre = int(batch.central_idx[graph])
        target = int(batch.target_idx[graph])
        records = torch.where(batch.record_mask[graph])[0].to(attention.device)
        background = records[records != target]
        target_weight = attention[graph, :, centre, target]
        background_weight = attention[graph, :, centre, background].mean(dim=-1)
        advantage[graph] = target_weight - background_weight
        ratio[graph] = target_weight / background_weight.clamp_min(1.0e-9)
    return {"attention_advantage": advantage, "attention_ratio": ratio}


def _valid_donor_mean(
    values: Sequence[torch.Tensor], valid: Sequence[torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    stacked = torch.stack(list(values), dim=0)
    validity = torch.stack(list(valid), dim=0).to(stacked.device)
    mean = finite_mean(stacked, validity, dim=0)
    return mean, validity.any(dim=0)


def _nan_invalid(value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    condition = valid.to(value.device)
    while condition.dim() < value.dim():
        condition = condition.unsqueeze(-1)
    return torch.where(condition, value, torch.full_like(value, float("nan")))


def intervention_layer_metrics(
    layer: DenseLayerCapture,
    phi: torch.Tensor,
    bundle: ReplicaBundle,
    batch: nar.NarBatch,
    intervention: str,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    clean_attention = layer.attention[0]
    clean_message = layer.message[0]
    route_values: list[torch.Tensor] = []
    message_values: list[torch.Tensor] = []
    retrieval_route_values: list[torch.Tensor] = []
    attention_delta_values: list[torch.Tensor] = []
    retrieval_attention_values: list[torch.Tensor] = []
    valid_values: list[torch.Tensor] = []
    max_absolute, max_relative = 0.0, 0.0

    retrieval_mask = torch.zeros(
        len(batch), batch.x.size(1), batch.x.size(1),
        dtype=torch.bool,
        device=clean_attention.device,
    )
    for graph in range(len(batch)):
        centre = int(batch.central_idx[graph])
        records = torch.where(batch.record_mask[graph])[0].to(clean_attention.device)
        retrieval_mask[graph, centre, records] = True

    for block, replica in bundle.items(intervention):
        route, message, total = symmetric_output_decomposition(
            clean_attention,
            clean_message,
            layer.attention[block],
            layer.message[block],
        )
        residual = total - route - message
        absolute = float(residual.abs().max().detach().cpu())
        scale = float(total.abs().max().detach().cpu())
        max_absolute = max(max_absolute, absolute)
        max_relative = max(max_relative, absolute / max(scale, EPS))
        route_values.append(route)
        message_values.append(message)
        valid_values.append(replica.valid)

        a_bar_message = 0.5 * (clean_message + layer.message[block])
        edge_route = (
            (clean_attention - layer.attention[block]).unsqueeze(-1) * a_bar_message
        )
        selected = retrieval_mask[:, None, :, :, None]
        retrieval_route_values.append(
            torch.where(selected, edge_route, torch.zeros_like(edge_route)).sum(dim=3)
        )
        attention_delta_values.append(clean_attention - layer.attention[block])
        selected_attention = torch.where(
            retrieval_mask[:, None],
            clean_attention - layer.attention[block],
            torch.zeros_like(clean_attention),
        )
        retrieval_attention_values.append(selected_attention)

    if not route_values:
        raise KeyError(f"replica bundle has no intervention {intervention!r}")
    mean_route, valid_any = _valid_donor_mean(route_values, valid_values)
    mean_message, _ = _valid_donor_mean(message_values, valid_values)
    mean_retrieval_route, _ = _valid_donor_mean(retrieval_route_values, valid_values)
    mean_attention_delta, _ = _valid_donor_mean(attention_delta_values, valid_values)
    mean_retrieval_attention, _ = _valid_donor_mean(
        retrieval_attention_values, valid_values
    )

    projected = project_components(phi, mean_route, mean_message)
    zero_message = torch.zeros_like(mean_retrieval_route)
    retrieval_projected = project_components(
        phi, mean_retrieval_route, zero_message
    )
    moved = 0.5 * mean_attention_delta.abs().sum(dim=(-1, -2))
    valid_edges = layer.mask[0].sum(dim=(-1, -2)).clamp_min(1)
    moved_normalized = moved / valid_edges
    retrieval_moved = 0.5 * mean_retrieval_attention.abs().sum(dim=(-1, -2))
    retrieval_edge_count = retrieval_mask.sum(dim=(-1, -2)).clamp_min(1)[:, None]
    retrieval_moved_normalized = retrieval_moved / retrieval_edge_count

    result = {
        key: _nan_invalid(value, valid_any)
        for key, value in projected.items()
        if not key.startswith("projected_")
    }
    result.update({
        "retrieval_route": _nan_invalid(retrieval_projected["route"], valid_any),
        "attention_moved": _nan_invalid(moved, valid_any),
        "attention_moved_normalized": _nan_invalid(moved_normalized, valid_any),
        "retrieval_attention_moved": _nan_invalid(retrieval_moved, valid_any),
        "retrieval_attention_moved_normalized": _nan_invalid(
            retrieval_moved_normalized, valid_any
        ),
        "valid": valid_any,
    })
    return result, {
        "closure_absolute": max_absolute,
        "closure_relative": max_relative,
    }


def intervention_output_metrics(
    captured: ForwardCapture,
    bundle: ReplicaBundle,
    intervention: str,
) -> dict[str, torch.Tensor]:
    clean = captured.logits[0]
    clean_labels = bundle.clean.y.to(clean.device).long()
    variant_logits: list[torch.Tensor] = []
    variant_labels: list[torch.Tensor] = []
    valid_values: list[torch.Tensor] = []
    for block, replica in bundle.items(intervention):
        variant_logits.append(captured.logits[block])
        variant_labels.append(replica.variant_label.to(clean.device).long())
        valid_values.append(replica.valid)
    mean_variant, valid_any = _valid_donor_mean(variant_logits, valid_values)
    functional = torch.linalg.vector_norm(clean - mean_variant, dim=-1)
    clean_loss = F.cross_entropy(clean, clean_labels, reduction="none")

    corrupt_losses = []
    counterfactual_losses = []
    counterfactual_correct = []
    for logits, labels in zip(variant_logits, variant_labels):
        corrupt_losses.append(F.cross_entropy(logits, clean_labels, reduction="none"))
        counterfactual_losses.append(F.cross_entropy(logits, labels, reduction="none"))
        counterfactual_correct.append((logits.argmax(dim=-1) == labels).float())
    corrupt_loss, _ = _valid_donor_mean(corrupt_losses, valid_values)
    counterfactual_loss, _ = _valid_donor_mean(counterfactual_losses, valid_values)
    counterfactual_accuracy, _ = _valid_donor_mean(
        counterfactual_correct, valid_values
    )
    return {
        "functional": _nan_invalid(functional, valid_any),
        "beneficial": _nan_invalid(corrupt_loss - clean_loss, valid_any),
        "counterfactual_loss": _nan_invalid(counterfactual_loss, valid_any),
        "counterfactual_accuracy": _nan_invalid(counterfactual_accuracy, valid_any),
        "valid": valid_any,
    }


def analyze_bundle(
    model: Any,
    bundle: ReplicaBundle,
    *,
    records: int,
    device: torch.device,
    closure_atol: float,
    closure_rtol: float,
    noop_tol: float,
) -> dict[str, Any]:
    combined = bundle.combined()
    blocks = 1 + len(bundle.replicas)
    graphs = len(bundle.clean)
    nodes = int(bundle.clean.x.size(1))
    captured = capture_analysis_forward(
        model,
        combined,
        graph_blocks=blocks,
        graphs_per_block=graphs,
        nodes=nodes,
        device=device,
    )
    interventions = sorted({item.intervention for item in bundle.replicas})
    head_metrics: dict[str, dict[str, list[torch.Tensor]]] = {
        intervention: defaultdict(list) for intervention in interventions
    }
    output_metrics = {
        intervention: intervention_output_metrics(captured, bundle, intervention)
        for intervention in interventions
    }
    clean_attention: dict[str, list[torch.Tensor]] = defaultdict(list)
    closure_absolute, closure_relative = 0.0, 0.0

    for layer_index, layer in enumerate(captured.layers):
        phi = batched_output_jacobian(
            captured.logits.reshape(blocks * graphs, records),
            layer.routed,
            clean_graphs=graphs,
            nodes=nodes,
            retain_graph=layer_index + 1 < len(captured.layers),
        )
        selection = clean_attention_selection(layer.attention[0], bundle.clean)
        for key, value in selection.items():
            clean_attention[key].append(value.detach().cpu())
        for intervention in interventions:
            metrics, closure = intervention_layer_metrics(
                layer, phi, bundle, bundle.clean, intervention
            )
            closure_absolute = max(closure_absolute, closure["closure_absolute"])
            closure_relative = max(closure_relative, closure["closure_relative"])
            for key, value in metrics.items():
                head_metrics[intervention][key].append(value.detach().cpu())

    if closure_absolute > closure_atol and closure_relative > closure_rtol:
        raise RuntimeError(
            "routing/message decomposition did not close: "
            f"absolute={closure_absolute:.3e}, relative={closure_relative:.3e}"
        )

    noop_errors = {}
    for intervention in (
        "identical",
        "record_permutation",
        "structural_record_record_noop",
    ):
        if intervention in output_metrics:
            value = output_metrics[intervention]["functional"]
            finite = value[torch.isfinite(value)]
            maximum = float(finite.max().detach().cpu()) if finite.numel() else float("nan")
            noop_errors[intervention] = maximum
            if math.isfinite(maximum) and maximum > noop_tol:
                raise RuntimeError(
                    f"{intervention} changed model output by {maximum:.3e} "
                    f"(tolerance {noop_tol:.3e})"
                )

    stacked_head = {
        intervention: {
            key: torch.stack(values, dim=1)
            for key, values in metrics.items()
        }
        for intervention, metrics in head_metrics.items()
    }
    stacked_clean = {
        key: torch.stack(values, dim=1) for key, values in clean_attention.items()
    }
    clean_logits = captured.logits[0].detach().cpu()
    clean_labels = bundle.clean.y.long()
    result = {
        "head": stacked_head,
        "clean_attention": stacked_clean,
        "output": {
            intervention: {key: value.detach().cpu() for key, value in metrics.items()}
            for intervention, metrics in output_metrics.items()
        },
        "clean": {
            "loss": F.cross_entropy(clean_logits, clean_labels, reduction="none"),
            "correct": (clean_logits.argmax(dim=-1) == clean_labels).float(),
            "logits": clean_logits,
            "labels": clean_labels,
        },
        "checks": {
            "closure_absolute": closure_absolute,
            "closure_relative": closure_relative,
            "noop_errors": noop_errors,
        },
    }
    return result


def _concat_nested_tensor_dict(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not items:
        return {}
    first = items[0]
    out: dict[str, Any] = {}
    for key, value in first.items():
        if torch.is_tensor(value):
            out[key] = torch.cat([item[key] for item in items], dim=0)
        elif isinstance(value, Mapping):
            out[key] = _concat_nested_tensor_dict([item[key] for item in items])
        else:
            out[key] = value
    return out


def adaptive_graph_chunk(
    cfg: AnalysisConfig,
    *,
    records: int,
    replica_blocks: int,
) -> int:
    nodes = int(records) + 3
    by_nodes = cfg.max_batch_nodes // max(1, nodes * replica_blocks)
    support_pairs = nodes * nodes
    by_pairs = cfg.max_replica_pairs // max(1, support_pairs * replica_blocks)
    return max(1, min(64, by_nodes, by_pairs))


def split_seed(
    cfg: AnalysisConfig,
    row: Mapping[str, Any],
    split: str,
) -> int:
    model_index = MODEL_ORDER.index(str(row["model"]))
    split_offset = {"discovery": 11, "estimation": 29, "causal": 47}[split]
    return (
        cfg.analysis_seed
        + int(row["seed"]) * 10_007
        + int(row["N"]) * 100_003
        + model_index * 1_000_003
        + split_offset * 10_000_019
    )


def run_metric_split(
    model: Any,
    nar_config: nar.Config,
    row: Mapping[str, Any],
    cfg: AnalysisConfig,
    *,
    split: str,
    graphs: int,
    device: torch.device,
) -> dict[str, Any]:
    records = int(row["N"])
    batch = nar.make_batch(
        nar_config,
        int(graphs),
        records,
        seed=split_seed(cfg, row, split),
    )
    probe = build_replica_bundle(
        batch.slice(0, min(2, len(batch))), records, cfg.donors
    )
    replica_checks = [verify_replica(probe.clean, item) for item in probe.replicas]
    replica_blocks = 1 + len(probe.replicas)
    chunk = adaptive_graph_chunk(
        cfg, records=records, replica_blocks=replica_blocks
    )
    pieces = []
    started = time.time()
    for start in range(0, len(batch), chunk):
        stop = min(start + chunk, len(batch))
        print(
            f"[metrics {row['model']} d={cfg.analysis_width} N={records} "
            f"seed={row['seed']} {split}] {start}:{stop}/{len(batch)}",
            flush=True,
        )
        part = batch.slice(start, stop)
        bundle = build_replica_bundle(part, records, cfg.donors)
        pieces.append(
            analyze_bundle(
                model,
                bundle,
                records=records,
                device=device,
                closure_atol=cfg.closure_atol,
                closure_rtol=cfg.closure_rtol,
                noop_tol=cfg.noop_tol,
            )
        )
        del bundle
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    combined = _concat_nested_tensor_dict(pieces)
    combined["checks"] = {
        "closure_absolute": max(piece["checks"]["closure_absolute"] for piece in pieces),
        "closure_relative": max(piece["checks"]["closure_relative"] for piece in pieces),
        "replicas": replica_checks,
        "noop_errors": {
            name: max(
                float(piece["checks"]["noop_errors"].get(name, float("nan")))
                for piece in pieces
            )
            for name in (
                "identical",
                "record_permutation",
                "structural_record_record_noop",
            )
        },
    }
    if str(row["model"]) == "1hop":
        address = combined["head"]["address_different_answer"]
        moved = address["retrieval_attention_moved"][:, 1]
        finite = moved[torch.isfinite(moved)]
        maximum = float(finite.max()) if finite.numel() else float("nan")
        combined["checks"]["onehop_layer2_address_routing_max"] = maximum
        if math.isfinite(maximum) and maximum > cfg.noop_tol:
            raise RuntimeError(
                "1-hop layer-2 centre-to-record attention changed under an address swap: "
                f"{maximum:.3e} > {cfg.noop_tol:.3e}"
            )
    combined["meta"] = {
        "split": split,
        "graphs": int(graphs),
        "seed": split_seed(cfg, row, split),
        "chunk_graphs": chunk,
        "replica_blocks": replica_blocks,
        "elapsed_s": time.time() - started,
    }
    return combined


def metric_cache_path(
    cfg: AnalysisConfig,
    row: Mapping[str, Any],
    sample_type: str,
) -> Path:
    stem = (
        f"{row['model']}__d{cfg.analysis_width}__N{row['N']}__seed_{row['seed']}"
        f"__{sample_type}__{scientific_fingerprint(cfg)}.pt"
    )
    return analysis_root(cfg) / "metrics" / stem


def analyze_checkpoint(
    row: Mapping[str, Any],
    cfg: AnalysisConfig,
    *,
    device: torch.device,
    force: bool,
) -> dict[str, Any]:
    sample_type = str(row["sample_type"])
    path = metric_cache_path(cfg, row, sample_type)
    if path.exists() and not force:
        cached = torch.load(path, map_location="cpu", weights_only=False)
        if cached.get("fingerprint") == scientific_fingerprint(cfg):
            print(f"[metrics cache] {path}", flush=True)
            return cached
    model, payload, nar_config = load_model_from_manifest(row, device)
    discovery = None
    if sample_type == "primary":
        discovery = run_metric_split(
            model,
            nar_config,
            row,
            cfg,
            split="discovery",
            graphs=cfg.discovery_graphs,
            device=device,
        )
    estimation = run_metric_split(
        model,
        nar_config,
        row,
        cfg,
        split="estimation",
        graphs=int(row["graphs"]),
        device=device,
    )
    result = {
        "analysis_version": ANALYSIS_VERSION,
        "fingerprint": scientific_fingerprint(cfg),
        "checkpoint": dict(row),
        "checkpoint_fingerprint": payload.get("fingerprint"),
        "sample_type": sample_type,
        "discovery": discovery,
        "estimation": estimation,
    }
    atomic_torch_save(path, result)
    print(f"[metrics saved] {path}", flush=True)
    del model
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def load_metric_caches(
    cfg: AnalysisConfig,
    schedule: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    out = []
    for row in schedule:
        path = metric_cache_path(cfg, row, str(row["sample_type"]))
        if not path.exists():
            raise FileNotFoundError(f"missing metric cache: {path}")
        cached = torch.load(path, map_location="cpu", weights_only=False)
        if cached.get("fingerprint") != scientific_fingerprint(cfg):
            raise RuntimeError(f"metric cache fingerprint mismatch: {path}")
        out.append(cached)
    return out
