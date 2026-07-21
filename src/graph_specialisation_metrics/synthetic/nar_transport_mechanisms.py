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
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from graph_specialisation_metrics.synthetic import nar_grit_fixed as nar


ANALYSIS_VERSION = "nar-transport-mechanisms-v2"
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
    structural_anchor_idx: torch.Tensor
    structural_partner_idx: torch.Tensor


@dataclass
class ReplicaBundle:
    clean: nar.NarBatch
    replicas: list[Replica]

    def combined(self) -> nar.NarBatch:
        combined = nar.concat_batches([self.clean] + [item.batch for item in self.replicas])
        empty = torch.full((len(self.clean), 2), -1, dtype=torch.long)
        swaps = [empty]
        for item in self.replicas:
            pair = torch.stack(
                [item.structural_anchor_idx.long(), item.structural_partner_idx.long()], dim=-1
            )
            valid = item.structural_partner_idx >= 0
            pair = torch.where(valid[:, None], pair, torch.full_like(pair, -1))
            swaps.append(pair)
        # NarBatch deliberately contains only task tensors. This analysis-only
        # attribute is copied explicitly in capture_analysis_forward.
        combined.structural_degree_swaps = torch.cat(swaps, dim=0)
        return combined

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
    for key in (
        "drive_root",
        "run_name",
        "grit_dir",
        "device",
        "bootstrap_samples",
        "max_batch_nodes",
        "max_dense_pairs",
        "max_replica_pairs",
    ):
        payload.pop(key, None)
    encoded = json.dumps(
        {"version": ANALYSIS_VERSION, **payload}, sort_keys=True, default=str
    ).encode()
    return hashlib.sha1(encoded).hexdigest()[:16]


def analysis_root(cfg: AnalysisConfig) -> Path:
    return (
        Path(cfg.drive_root)
        / cfg.run_name
        / "transport_mechanisms_v2"
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
    selection_scope = {
        "width": cfg.analysis_width,
        "models": list(cfg.models),
        "ns": list(cfg.ns),
        "seeds": list(cfg.seeds),
    }
    if json_path.exists() and not force:
        cached = json.loads(json_path.read_text(encoding="utf-8"))
        if (
            cached.get("analysis_version") == ANALYSIS_VERSION
            and cached.get("selection_scope") == selection_scope
        ):
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
            "selection_scope": selection_scope,
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
        structural_anchor_idx=torch.full_like(target, -1),
        structural_partner_idx=torch.full_like(target, -1),
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
        item.structural_anchor_idx[graph] = query
        item.structural_partner_idx[graph] = partner
    return item


def structural_record_record_noop(batch: nar.NarBatch, donor: int) -> Replica:
    item = _base_replica("structural_record_record_noop", donor, batch)
    for graph in range(len(batch)):
        records = torch.where(batch.record_mask[graph])[0].tolist()
        left = int(records[int(donor) % len(records)])
        right = int(records[(int(donor) + 1) % len(records)])
        item.batch.rrwp[graph] = _swap_rrwp_role(batch.rrwp[graph], left, right)
        item.structural_anchor_idx[graph] = left
        item.structural_partner_idx[graph] = right
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
        if (
            not torch.equal(replica.batch.query_idx, clean.query_idx)
            or not torch.equal(replica.batch.target_idx, clean.target_idx)
            or not torch.equal(replica.batch.y, clean.y)
        ):
            raise RuntimeError(f"{name} changed task metadata")
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
    if hasattr(combined, "structural_degree_swaps"):
        moved.structural_degree_swaps = combined.structural_degree_swaps.to(device)
    original_pyg_batch = model._pyg_batch

    def structurally_aware_pyg_batch(batch: nar.NarBatch) -> Any:
        data = original_pyg_batch(batch)
        swaps = getattr(batch, "structural_degree_swaps", None)
        if swaps is None or not bool((swaps[:, 0] >= 0).any()):
            return data
        graph_count = int(batch.x.size(0))
        node_count = int(batch.x.size(1))
        degree = data.deg.reshape(graph_count, node_count).clone()
        for graph in torch.where(swaps[:, 0] >= 0)[0].tolist():
            left = int(swaps[graph, 0])
            right = int(swaps[graph, 1])
            left_value = degree[graph, left].clone()
            degree[graph, left] = degree[graph, right]
            degree[graph, right] = left_value
        data.deg = degree.reshape(-1)
        data.log_deg = torch.log(data.deg + 1.0)
        return data

    model._pyg_batch = structurally_aware_pyg_batch
    try:
        with OfficialGRITMechanisticCollector(model) as collector:
            logits = model(moved)
    finally:
        model._pyg_batch = original_pyg_batch
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


def retrieval_attention_diagnostics(
    clean_attention: torch.Tensor,
    variant_attention: torch.Tensor,
    retrieval_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Separate record-channel gating from within-record routing selection.

    ``clean_attention`` and ``variant_attention`` are ``[B,H,N,N]`` routing
    weights and ``retrieval_mask`` selects centre-to-record pairs.  A changed
    competing non-record logit can rescale every selected attention value through
    the softmax denominator without changing the relative distribution over
    records.  The raw response therefore measures record-channel gating plus
    selection, whereas ``profile_moved`` isolates address-selective reweighting.
    """

    selected = retrieval_mask[:, None]
    clean_selected = torch.where(
        selected, clean_attention, torch.zeros_like(clean_attention)
    )
    variant_selected = torch.where(
        selected, variant_attention, torch.zeros_like(variant_attention)
    )
    clean_mass = clean_selected.sum(dim=(-1, -2))
    variant_mass = variant_selected.sum(dim=(-1, -2))
    clean_profile = clean_selected / clean_mass.clamp_min(EPS)[..., None, None]
    variant_profile = (
        variant_selected / variant_mass.clamp_min(EPS)[..., None, None]
    )
    profile_valid = (clean_mass > EPS) & (variant_mass > EPS)
    profile_moved = 0.5 * (clean_profile - variant_profile).abs().sum(dim=(-1, -2))
    profile_moved = torch.where(
        profile_valid,
        profile_moved,
        torch.full_like(profile_moved, float("nan")),
    )
    return {
        "raw_moved": 0.5 * (clean_selected - variant_selected).abs().sum(
            dim=(-1, -2)
        ),
        "gate_moved": (clean_mass - variant_mass).abs(),
        "profile_moved": profile_moved,
    }


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
    retrieval_gate_values: list[torch.Tensor] = []
    retrieval_profile_values: list[torch.Tensor] = []
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
        retrieval_diagnostics = retrieval_attention_diagnostics(
            clean_attention,
            layer.attention[block],
            retrieval_mask,
        )
        retrieval_gate_values.append(retrieval_diagnostics["gate_moved"])
        retrieval_profile_values.append(retrieval_diagnostics["profile_moved"])

    if not route_values:
        raise KeyError(f"replica bundle has no intervention {intervention!r}")
    mean_route, valid_any = _valid_donor_mean(route_values, valid_values)
    mean_message, _ = _valid_donor_mean(message_values, valid_values)
    mean_retrieval_route, _ = _valid_donor_mean(retrieval_route_values, valid_values)
    mean_attention_delta, _ = _valid_donor_mean(attention_delta_values, valid_values)
    mean_retrieval_attention, _ = _valid_donor_mean(
        retrieval_attention_values, valid_values
    )
    mean_retrieval_gate, _ = _valid_donor_mean(
        retrieval_gate_values, valid_values
    )
    mean_retrieval_profile, _ = _valid_donor_mean(
        retrieval_profile_values, valid_values
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
        "retrieval_attention_gate_moved": _nan_invalid(
            mean_retrieval_gate, valid_any
        ),
        "retrieval_attention_profile_moved": _nan_invalid(
            mean_retrieval_profile, valid_any
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
        routed_clean = layer.routed[: graphs * nodes].reshape(
            graphs, nodes, model.H, model.dh
        )
        clean_throughput = torch.linalg.vector_norm(
            routed_clean.float(), dim=-1
        ).sum(dim=1)
        clean_attention["clean_throughput"].append(clean_throughput.detach().cpu())
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
    force: bool = False,
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
        chunk_path = metric_chunk_cache_path(
            cfg, row, str(row["sample_type"]), split, start, stop
        )
        if chunk_path.exists() and not force:
            cached = torch.load(chunk_path, map_location="cpu", weights_only=False)
            if cached.get("fingerprint") == scientific_fingerprint(cfg):
                print(f"[metric chunk cache] {chunk_path}", flush=True)
                pieces.append(cached["result"])
                continue
        print(
            f"[metrics {row['model']} d={cfg.analysis_width} N={records} "
            f"seed={row['seed']} {split}] {start}:{stop}/{len(batch)}",
            flush=True,
        )
        part = batch.slice(start, stop)
        bundle = build_replica_bundle(part, records, cfg.donors)
        piece = analyze_bundle(
            model,
            bundle,
            records=records,
            device=device,
            closure_atol=cfg.closure_atol,
            closure_rtol=cfg.closure_rtol,
            noop_tol=cfg.noop_tol,
        )
        atomic_torch_save(chunk_path, {
            "analysis_version": ANALYSIS_VERSION,
            "fingerprint": scientific_fingerprint(cfg),
            "checkpoint": dict(row),
            "sample_type": str(row["sample_type"]),
            "split": split,
            "start": start,
            "stop": stop,
            "result": piece,
        })
        print(f"[metric chunk saved] {chunk_path}", flush=True)
        pieces.append(piece)
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
        for source, destination in (
            (
                "retrieval_attention_moved",
                "onehop_layer2_address_raw_attention_moved_max",
            ),
            (
                "retrieval_attention_gate_moved",
                "onehop_layer2_address_gate_moved_max",
            ),
            (
                "retrieval_attention_profile_moved",
                "onehop_layer2_address_profile_moved_max",
            ),
        ):
            values = address[source][:, 1]
            finite = values[torch.isfinite(values)]
            combined["checks"][destination] = (
                float(finite.max()) if finite.numel() else float("nan")
            )
        profile_max = combined["checks"][
            "onehop_layer2_address_profile_moved_max"
        ]
        combined["checks"]["onehop_layer2_address_profile_zero_pass"] = bool(
            math.isfinite(profile_max) and profile_max <= cfg.noop_tol
        )
        if math.isfinite(profile_max) and profile_max > cfg.noop_tol:
            print(
                "[diagnostic warning] 1-hop layer-2 within-record attention profile "
                f"changed under an address swap: {profile_max:.3e} > "
                f"{cfg.noop_tol:.3e}. Results will be cached and the verification "
                "check marked failed.",
                flush=True,
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


def metric_chunk_cache_path(
    cfg: AnalysisConfig,
    row: Mapping[str, Any],
    sample_type: str,
    split: str,
    start: int,
    stop: int,
) -> Path:
    stem = (
        f"{row['model']}__d{cfg.analysis_width}__N{row['N']}__seed_{row['seed']}"
        f"__{sample_type}__{split}__graphs_{start}_{stop}"
        f"__{scientific_fingerprint(cfg)}.pt"
    )
    return analysis_root(cfg) / "metrics" / "chunks" / stem


def metric_split_cache_path(
    cfg: AnalysisConfig,
    row: Mapping[str, Any],
    sample_type: str,
    split: str,
) -> Path:
    stem = (
        f"{row['model']}__d{cfg.analysis_width}__N{row['N']}__seed_{row['seed']}"
        f"__{sample_type}__{split}__{scientific_fingerprint(cfg)}.pt"
    )
    return analysis_root(cfg) / "metrics" / "splits" / stem


def load_or_run_metric_split(
    model: Any,
    nar_config: nar.Config,
    row: Mapping[str, Any],
    cfg: AnalysisConfig,
    *,
    sample_type: str,
    split: str,
    graphs: int,
    device: torch.device,
    force: bool,
) -> dict[str, Any]:
    path = metric_split_cache_path(cfg, row, sample_type, split)
    if path.exists() and not force:
        cached = torch.load(path, map_location="cpu", weights_only=False)
        if cached.get("fingerprint") == scientific_fingerprint(cfg):
            print(f"[metric split cache] {path}", flush=True)
            return cached["result"]
    result = run_metric_split(
        model,
        nar_config,
        row,
        cfg,
        split=split,
        graphs=graphs,
        device=device,
        force=force,
    )
    atomic_torch_save(path, {
        "analysis_version": ANALYSIS_VERSION,
        "fingerprint": scientific_fingerprint(cfg),
        "checkpoint": dict(row),
        "sample_type": sample_type,
        "split": split,
        "result": result,
    })
    print(f"[metric split saved] {path}", flush=True)
    return result


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
        discovery = load_or_run_metric_split(
            model,
            nar_config,
            row,
            cfg,
            sample_type=sample_type,
            split="discovery",
            graphs=cfg.discovery_graphs,
            device=device,
            force=force,
        )
    estimation = load_or_run_metric_split(
        model,
        nar_config,
        row,
        cfg,
        sample_type=sample_type,
        split="estimation",
        graphs=int(row["graphs"]),
        device=device,
        force=force,
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


def decode_head(index: int, heads: int) -> tuple[int, int]:
    return int(index // heads), int(index % heads)


def _sample_layer_matched_families(
    target: Sequence[tuple[int, int]],
    throughput: torch.Tensor,
    *,
    count: int,
    seed: int,
) -> list[list[tuple[int, int]]]:
    rng = np.random.default_rng(int(seed))
    layers, heads = int(throughput.size(0)), int(throughput.size(1))
    excluded = set((int(layer), int(head)) for layer, head in target)
    layer_counts: dict[int, int] = defaultdict(int)
    for layer, _head in target:
        layer_counts[int(layer)] += 1
    target_throughput = sum(float(throughput[layer, head]) for layer, head in target)
    candidates: dict[tuple[tuple[int, int], ...], float] = {}
    attempts = max(256, int(count) * 128)
    for _ in range(attempts):
        family: list[tuple[int, int]] = []
        valid = True
        for layer in range(layers):
            needed = int(layer_counts.get(layer, 0))
            if not needed:
                continue
            pool = [head for head in range(heads) if (layer, head) not in excluded]
            if len(pool) < needed:
                valid = False
                break
            chosen = rng.choice(pool, size=needed, replace=False).tolist()
            family.extend((layer, int(head)) for head in chosen)
        if not valid:
            continue
        key = tuple(sorted(family))
        value = sum(float(throughput[layer, head]) for layer, head in key)
        candidates[key] = abs(math.log((value + EPS) / (target_throughput + EPS)))
    if len(candidates) < int(count):
        raise RuntimeError(
            f"could not construct {count} layer-matched random families for {target}"
        )
    ordered = sorted(candidates, key=lambda key: (candidates[key], key))[: int(count)]
    return [list(item) for item in ordered]


def select_causal_families(
    discovery: Mapping[str, Any],
    cfg: AnalysisConfig,
    row: Mapping[str, Any],
) -> dict[str, Any]:
    address_route = torch.nanmean(
        discovery["head"]["address_different_answer"]["route"], dim=0
    )
    payload_message = torch.nanmean(
        discovery["head"]["target_payload"]["message"], dim=0
    )
    throughput = torch.nanmean(discovery["clean_attention"]["clean_throughput"], dim=0)
    heads = int(address_route.size(1))
    size = min(int(cfg.family_size), int(address_route.numel()))
    routing_indices = torch.argsort(address_route.flatten(), descending=True)[:size].tolist()
    message_indices = torch.argsort(payload_message.flatten(), descending=True)[:size].tolist()
    routing = [decode_head(index, heads) for index in routing_indices]
    message = [decode_head(index, heads) for index in message_indices]
    base_seed = (
        cfg.analysis_seed
        + int(row["seed"]) * 10_007
        + int(row["N"]) * 100_003
        + MODEL_ORDER.index(str(row["model"])) * 1_000_003
    )
    routing_controls = _sample_layer_matched_families(
        routing,
        throughput,
        count=cfg.random_families,
        seed=base_seed + 101,
    )
    message_controls = _sample_layer_matched_families(
        message,
        throughput,
        count=cfg.random_families,
        seed=base_seed + 211,
    )
    return {
        "routing": routing,
        "message": message,
        "routing_controls": routing_controls,
        "message_controls": message_controls,
        "overlap": sorted(set(routing) & set(message)),
        "routing_scores": address_route,
        "message_scores": payload_message,
        "clean_throughput": throughput,
    }


@dataclass
class SparseEndpoint:
    layer: int
    graph: torch.Tensor
    local_source: torch.Tensor
    local_destination: torch.Tensor
    attention: torch.Tensor
    message: torch.Tensor


class EndpointCollector:
    def __init__(self, model: Any) -> None:
        self.model = model
        self.records: dict[int, SparseEndpoint] = {}
        self.handles: list[Any] = []

    def __enter__(self) -> "EndpointCollector":
        from graph_specialisation_metrics import mechanistic_operator_analysis as moa

        def make_hook(layer_index: int):
            def hook(module: Any, inputs: tuple[Any, ...], _outputs: Any) -> None:
                pyg_batch = inputs[0]
                node_message, pair_message, _logits, _edge_state = (
                    moa.grit_attention_components(module, pyg_batch)
                )
                edge_index = pyg_batch.edge_index.long()
                graph, local_source, local_destination = moa.edge_local_coordinates(
                    edge_index[0],
                    edge_index[1],
                    [
                        int(value)
                        for value in pyg_batch.graph_num_nodes.detach().cpu().tolist()
                    ],
                )
                self.records[layer_index] = SparseEndpoint(
                    layer=layer_index,
                    graph=graph.detach(),
                    local_source=local_source.detach(),
                    local_destination=local_destination.detach(),
                    attention=pyg_batch.attn.squeeze(-1).detach().float(),
                    message=(node_message + pair_message).detach().float(),
                )

            return hook

        self.records = {}
        self.handles = [
            layer.attention.register_forward_hook(make_hook(layer_index))
            for layer_index, layer in enumerate(self.model.layers)
        ]
        return self

    def __exit__(self, *_exc: object) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []


@torch.no_grad()
def capture_endpoint_forward(
    model: Any,
    batch: nar.NarBatch,
    device: torch.device,
) -> tuple[torch.Tensor, dict[int, SparseEndpoint]]:
    model.eval()
    moved = batch.to(device)
    with EndpointCollector(model) as collector:
        logits = model(moved)
    if len(collector.records) != int(model.L):
        raise RuntimeError("failed to capture every GRIT layer endpoint")
    return logits.detach(), dict(collector.records)


def _endpoint_keys(endpoint: SparseEndpoint, nodes: int) -> torch.Tensor:
    return (
        endpoint.graph.long() * int(nodes) * int(nodes)
        + endpoint.local_destination.long() * int(nodes)
        + endpoint.local_source.long()
    )


def align_endpoint(
    source: SparseEndpoint,
    graph: torch.Tensor,
    local_source: torch.Tensor,
    local_destination: torch.Tensor,
    nodes: int,
) -> torch.Tensor:
    source_keys = _endpoint_keys(source, nodes)
    current_keys = (
        graph.long() * int(nodes) * int(nodes)
        + local_destination.long() * int(nodes)
        + local_source.long()
    )
    order = torch.argsort(source_keys)
    sorted_keys = source_keys[order]
    positions = torch.searchsorted(sorted_keys, current_keys)
    clamped = positions.clamp(max=max(int(sorted_keys.numel()) - 1, 0))
    found = (positions < sorted_keys.numel()) & (sorted_keys[clamped] == current_keys)
    if not bool(found.all()):
        raise RuntimeError(
            f"failed to align {int((~found).sum())} clean/variant sparse edges"
        )
    return order[clamped]


class HybridHeadPatch:
    """Patch clean routing and/or messages into one variant GRIT layer."""

    def __init__(
        self,
        model: Any,
        *,
        layer: int,
        heads: Sequence[int],
        clean: SparseEndpoint,
        nodes: int,
        clean_routing: bool,
        clean_message: bool,
    ) -> None:
        self.model = model
        self.layer = int(layer)
        self.heads = tuple(sorted(set(map(int, heads))))
        self.clean = clean
        self.nodes = int(nodes)
        self.clean_routing = bool(clean_routing)
        self.clean_message = bool(clean_message)
        self.original: Any = None

    def __enter__(self) -> "HybridHeadPatch":
        attention = self.model.layers[self.layer].attention
        self.original = attention.propagate_attention
        attention.propagate_attention = self._make_propagate(attention)
        return self

    def __exit__(self, *_exc: object) -> None:
        attention = self.model.layers[self.layer].attention
        if self.original is not None:
            attention.propagate_attention = self.original
        self.original = None

    def _make_propagate(self, attention_module: Any):
        def propagate(pyg_batch: Any) -> None:
            from torch_scatter import scatter
            from graph_specialisation_metrics import mechanistic_operator_analysis as moa

            edge_index = pyg_batch.edge_index.long()
            graph, local_source, local_destination = moa.edge_local_coordinates(
                edge_index[0],
                edge_index[1],
                [
                    int(value)
                    for value in pyg_batch.graph_num_nodes.detach().cpu().tolist()
                ],
            )
            positions = align_endpoint(
                self.clean,
                graph,
                local_source,
                local_destination,
                self.nodes,
            ).to(edge_index.device)
            node_message, pair_message, logits, edge_state = moa.grit_attention_components(
                attention_module, pyg_batch
            )
            score = moa.pyg_sparse_softmax(
                logits.unsqueeze(-1), edge_index[1], pyg_batch.num_nodes
            ).squeeze(-1)
            message = node_message + pair_message
            head_index = torch.tensor(self.heads, dtype=torch.long, device=edge_index.device)
            if self.clean_routing:
                replacement = self.clean.attention.to(score.device, score.dtype)[positions]
                score = score.clone()
                score[:, head_index] = replacement[:, head_index]
            if self.clean_message:
                replacement = self.clean.message.to(message.device, message.dtype)[positions]
                message = message.clone()
                message[:, head_index] = replacement[:, head_index]
            score = attention_module.dropout(score.unsqueeze(-1))
            pyg_batch.attn = score
            if getattr(pyg_batch, "E", None) is not None:
                pyg_batch.wE = edge_state.flatten(1)
            weighted = message * score
            pyg_batch.wV = torch.zeros_like(pyg_batch.V_h)
            scatter(weighted, edge_index[1], dim=0, out=pyg_batch.wV, reduce="add")

        return propagate


@torch.no_grad()
def predict_hybrid(
    model: Any,
    batch: nar.NarBatch,
    *,
    device: torch.device,
    layer: int,
    heads: Sequence[int],
    clean_endpoint: SparseEndpoint,
    clean_routing: bool,
    clean_message: bool,
) -> torch.Tensor:
    context = HybridHeadPatch(
        model,
        layer=layer,
        heads=heads,
        clean=clean_endpoint,
        nodes=int(batch.x.size(1)),
        clean_routing=clean_routing,
        clean_message=clean_message,
    )
    with context:
        return model(batch.to(device)).detach().cpu()


def _log_probability(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.log_softmax(logits.float(), dim=-1).gather(
        1, labels.long().view(-1, 1)
    ).squeeze(1)


def _family_entries(families: Mapping[str, Any]) -> list[dict[str, Any]]:
    entries = [
        {"family": "routing", "replicate": 0, "heads": list(families["routing"])},
        {"family": "message", "replicate": 0, "heads": list(families["message"])},
    ]
    for family in ("routing", "message"):
        for replicate, heads in enumerate(families[f"{family}_controls"]):
            entries.append({
                "family": f"{family}_random",
                "replicate": replicate,
                "heads": list(heads),
            })
    return entries


def _append_ablation_rows(
    rows: list[dict[str, Any]],
    *,
    model: Any,
    batch: nar.NarBatch,
    baseline: torch.Tensor,
    family: Mapping[str, Any],
    device: torch.device,
    graph_offset: int,
) -> None:
    labels = batch.y.long()
    baseline_cpu = baseline.detach().cpu()
    base_loss = F.cross_entropy(baseline_cpu, labels, reduction="none")
    with nar.ablate_heads(model, family["heads"]), torch.no_grad():
        changed = model(batch.to(device)).detach().cpu()
    changed_loss = F.cross_entropy(changed, labels, reduction="none")
    for graph in range(len(batch)):
        rows.append({
            "graph": graph_offset + graph,
            "family": family["family"],
            "replicate": int(family["replicate"]),
            "heads": str(family["heads"]),
            "loss_delta": float(changed_loss[graph] - base_loss[graph]),
            "accuracy_drop": float(
                (baseline_cpu[graph].argmax() == labels[graph]).float()
                - (changed[graph].argmax() == labels[graph]).float()
            ),
            "logit_displacement": float(
                torch.linalg.vector_norm(changed[graph] - baseline_cpu[graph])
            ),
        })


def _append_patch_rows(
    rows: list[dict[str, Any]],
    *,
    intervention: str,
    donor: int,
    replica: Replica,
    variant_logits: torch.Tensor,
    clean_endpoints: Mapping[int, SparseEndpoint],
    family: Mapping[str, Any],
    model: Any,
    device: torch.device,
    graph_offset: int,
) -> None:
    by_layer: dict[int, list[int]] = defaultdict(list)
    for layer, head in family["heads"]:
        by_layer[int(layer)].append(int(head))
    clean_labels = replica.clean_label.long()
    variant_labels = replica.variant_label.long()
    vv = variant_logits.detach().cpu()
    for layer, heads in by_layer.items():
        cv = predict_hybrid(
            model,
            replica.batch,
            device=device,
            layer=layer,
            heads=heads,
            clean_endpoint=clean_endpoints[layer],
            clean_routing=True,
            clean_message=False,
        )
        vc = predict_hybrid(
            model,
            replica.batch,
            device=device,
            layer=layer,
            heads=heads,
            clean_endpoint=clean_endpoints[layer],
            clean_routing=False,
            clean_message=True,
        )
        cc = predict_hybrid(
            model,
            replica.batch,
            device=device,
            layer=layer,
            heads=heads,
            clean_endpoint=clean_endpoints[layer],
            clean_routing=True,
            clean_message=True,
        )
        f_vv = _log_probability(vv, clean_labels)
        f_cv = _log_probability(cv, clean_labels)
        f_vc = _log_probability(vc, clean_labels)
        f_cc = _log_probability(cc, clean_labels)
        new_vv = _log_probability(vv, variant_labels)
        new_cv = _log_probability(cv, variant_labels)
        new_vc = _log_probability(vc, variant_labels)
        new_cc = _log_probability(cc, variant_labels)
        for graph in range(len(replica.batch)):
            if not bool(replica.valid[graph]):
                continue
            rows.append({
                "graph": graph_offset + graph,
                "intervention": intervention,
                "donor": int(donor),
                "family": family["family"],
                "replicate": int(family["replicate"]),
                "heads": str(family["heads"]),
                "layer": int(layer),
                "routing_rescue": float(f_cv[graph] - f_vv[graph]),
                "message_rescue": float(f_vc[graph] - f_vv[graph]),
                "full_rescue": float(f_cc[graph] - f_vv[graph]),
                "interaction": float(
                    f_cc[graph] - f_cv[graph] - f_vc[graph] + f_vv[graph]
                ),
                "new_target_routing_change": float(new_cv[graph] - new_vv[graph]),
                "new_target_message_change": float(new_vc[graph] - new_vv[graph]),
                "new_target_full_change": float(new_cc[graph] - new_vv[graph]),
                "variant_clean_logp": float(f_vv[graph]),
                "variant_new_logp": float(new_vv[graph]),
            })


def causal_cache_path(cfg: AnalysisConfig, row: Mapping[str, Any]) -> Path:
    stem = (
        f"{row['model']}__d{cfg.analysis_width}__N{row['N']}__seed_{row['seed']}"
        f"__{scientific_fingerprint(cfg)}.pt"
    )
    return analysis_root(cfg) / "causal" / stem


def causal_chunk_cache_path(
    cfg: AnalysisConfig,
    row: Mapping[str, Any],
    start: int,
    stop: int,
) -> Path:
    stem = (
        f"{row['model']}__d{cfg.analysis_width}__N{row['N']}__seed_{row['seed']}"
        f"__graphs_{start}_{stop}__{scientific_fingerprint(cfg)}.pt"
    )
    return analysis_root(cfg) / "causal" / "chunks" / stem


def run_causal_checkpoint(
    row: Mapping[str, Any],
    metric_cache: Mapping[str, Any],
    cfg: AnalysisConfig,
    *,
    device: torch.device,
    force: bool,
) -> dict[str, Any]:
    path = causal_cache_path(cfg, row)
    if path.exists() and not force:
        cached = torch.load(path, map_location="cpu", weights_only=False)
        if cached.get("fingerprint") == scientific_fingerprint(cfg):
            print(f"[causal cache] {path}", flush=True)
            return cached
    discovery = metric_cache.get("discovery")
    if discovery is None:
        raise RuntimeError("causal family selection requires the primary discovery cache")
    families = select_causal_families(discovery, cfg, row)
    model, _payload, nar_config = load_model_from_manifest(row, device)
    records = int(row["N"])
    full_batch = nar.make_batch(
        nar_config,
        cfg.causal_graphs,
        records,
        seed=split_seed(cfg, row, "causal"),
    )
    node_count = int(full_batch.x.size(1))
    chunk = max(1, min(
        64,
        cfg.max_batch_nodes // node_count,
        cfg.max_dense_pairs // (node_count * node_count),
    ))
    ablation_rows: list[dict[str, Any]] = []
    patch_rows: list[dict[str, Any]] = []
    entries = _family_entries(families)
    # Patching every matched-null replicate is needlessly expensive. Two per
    # target family provide a finite causal null; all controls are retained for ablation.
    patch_entries = [
        entry
        for entry in entries
        if not entry["family"].endswith("_random") or int(entry["replicate"]) < 2
    ]
    started = time.time()
    for start in range(0, len(full_batch), chunk):
        stop = min(start + chunk, len(full_batch))
        chunk_path = causal_chunk_cache_path(cfg, row, start, stop)
        if chunk_path.exists() and not force:
            chunk_cached = torch.load(chunk_path, map_location="cpu", weights_only=False)
            if chunk_cached.get("fingerprint") == scientific_fingerprint(cfg):
                print(f"[causal chunk cache] {chunk_path}", flush=True)
                ablation_rows.extend(chunk_cached["ablation_rows"])
                patch_rows.extend(chunk_cached["patch_rows"])
                continue
        print(
            f"[causal {row['model']} d={cfg.analysis_width} N={records} seed={row['seed']}] "
            f"{start}:{stop}/{len(full_batch)}",
            flush=True,
        )
        clean = full_batch.slice(start, stop)
        clean_logits, clean_endpoints = capture_endpoint_forward(model, clean, device)
        chunk_ablation_rows: list[dict[str, Any]] = []
        chunk_patch_rows: list[dict[str, Any]] = []
        for family in entries:
            _append_ablation_rows(
                chunk_ablation_rows,
                model=model,
                batch=clean,
                baseline=clean_logits,
                family=family,
                device=device,
                graph_offset=start,
            )
        for donor in range(cfg.causal_donors):
            replicas = (
                target_payload_replica(clean, records, donor),
                address_replica(clean, records, donor, same_answer=False),
            )
            for replica in replicas:
                variant_logits, _variant_endpoints = capture_endpoint_forward(
                    model, replica.batch, device
                )
                for family in patch_entries:
                    _append_patch_rows(
                        chunk_patch_rows,
                        intervention=replica.intervention,
                        donor=donor,
                        replica=replica,
                        variant_logits=variant_logits,
                        clean_endpoints=clean_endpoints,
                        family=family,
                        model=model,
                        device=device,
                        graph_offset=start,
                    )
        atomic_torch_save(chunk_path, {
            "analysis_version": ANALYSIS_VERSION,
            "fingerprint": scientific_fingerprint(cfg),
            "checkpoint": dict(row),
            "start": start,
            "stop": stop,
            "families": families,
            "ablation_rows": chunk_ablation_rows,
            "patch_rows": chunk_patch_rows,
        })
        print(f"[causal chunk saved] {chunk_path}", flush=True)
        ablation_rows.extend(chunk_ablation_rows)
        patch_rows.extend(chunk_patch_rows)
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    result = {
        "analysis_version": ANALYSIS_VERSION,
        "fingerprint": scientific_fingerprint(cfg),
        "checkpoint": dict(row),
        "families": families,
        "ablation_rows": ablation_rows,
        "patch_rows": patch_rows,
        "meta": {
            "graphs": cfg.causal_graphs,
            "donors": cfg.causal_donors,
            "seed": split_seed(cfg, row, "causal"),
            "chunk_graphs": chunk,
            "elapsed_s": time.time() - started,
        },
    }
    atomic_torch_save(path, result)
    print(f"[causal saved] {path}", flush=True)
    del model
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def causal_schedule(
    manifest: Sequence[Mapping[str, Any]], cfg: AnalysisConfig
) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in manifest
        if bool(row["selected_for_analysis"]) and int(row["N"]) in cfg.anchor_ns
    ]


def load_causal_caches(
    cfg: AnalysisConfig,
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        path = causal_cache_path(cfg, row)
        if not path.exists():
            raise FileNotFoundError(f"missing causal cache: {path}")
        cached = torch.load(path, map_location="cpu", weights_only=False)
        if cached.get("fingerprint") != scientific_fingerprint(cfg):
            raise RuntimeError(f"causal cache fingerprint mismatch: {path}")
        out.append(cached)
    return out


def _tensor_mean(value: torch.Tensor) -> float:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    finite = torch.isfinite(tensor)
    return float(tensor[finite].mean()) if finite.any() else float("nan")


def build_metric_tables(
    caches: Sequence[Mapping[str, Any]],
    cfg: AnalysisConfig,
) -> dict[str, list[dict[str, Any]]]:
    head_rows: list[dict[str, Any]] = []
    output_rows: list[dict[str, Any]] = []
    correlation_rows: list[dict[str, Any]] = []
    check_rows: list[dict[str, Any]] = []
    context_rows: list[dict[str, Any]] = []

    for cached in caches:
        checkpoint = cached["checkpoint"]
        estimation = cached["estimation"]
        base = {
            "model": checkpoint["model"],
            "width": int(checkpoint["width"]),
            "N": int(checkpoint["N"]),
            "seed": int(checkpoint["seed"]),
            "sample_type": cached["sample_type"],
            "selected": bool(checkpoint["selected_for_analysis"]),
            "heldout_accuracy": float(checkpoint["heldout_accuracy"]),
        }
        clean_attention = estimation["clean_attention"]
        for intervention, metrics in estimation["head"].items():
            route = metrics["route"]
            graphs, layers, heads = map(int, route.shape)
            for layer in range(layers):
                for head in range(heads):
                    row = {
                        **base,
                        "intervention": intervention,
                        "layer": layer,
                        "head": head,
                        "graphs": graphs,
                        "route": _tensor_mean(metrics["route"][:, layer, head]),
                        "message": _tensor_mean(metrics["message"][:, layer, head]),
                        "total": _tensor_mean(metrics["total"][:, layer, head]),
                        "routing_share": _tensor_mean(
                            metrics["routing_share"][:, layer, head]
                        ),
                        "mechanism_balance": _tensor_mean(
                            metrics["mechanism_balance"][:, layer, head]
                        ),
                        "alignment": _tensor_mean(metrics["alignment"][:, layer, head]),
                        "retrieval_route": _tensor_mean(
                            metrics["retrieval_route"][:, layer, head]
                        ),
                        "attention_moved": _tensor_mean(
                            metrics["attention_moved"][:, layer, head]
                        ),
                        "attention_moved_normalized": _tensor_mean(
                            metrics["attention_moved_normalized"][:, layer, head]
                        ),
                        "retrieval_attention_moved": _tensor_mean(
                            metrics["retrieval_attention_moved"][:, layer, head]
                        ),
                        "retrieval_attention_moved_normalized": _tensor_mean(
                            metrics["retrieval_attention_moved_normalized"][:, layer, head]
                        ),
                        "retrieval_attention_gate_moved": _tensor_mean(
                            metrics["retrieval_attention_gate_moved"][:, layer, head]
                        ),
                        "retrieval_attention_profile_moved": _tensor_mean(
                            metrics["retrieval_attention_profile_moved"][:, layer, head]
                        ),
                        "attention_advantage": _tensor_mean(
                            clean_attention["attention_advantage"][:, layer, head]
                        ),
                        "attention_ratio": _tensor_mean(
                            clean_attention["attention_ratio"][:, layer, head]
                        ),
                        "clean_throughput": _tensor_mean(
                            clean_attention["clean_throughput"][:, layer, head]
                        ),
                    }
                    head_rows.append(row)
            flat_total = torch.nanmean(metrics["total"], dim=0).flatten().numpy()
            flat_moved = torch.nanmean(
                metrics["attention_moved_normalized"], dim=0
            ).flatten().numpy()
            flat_clean = torch.nanmean(
                clean_attention["attention_ratio"], dim=0
            ).flatten().numpy()
            flat_route = torch.nanmean(metrics["route"], dim=0).flatten().numpy()
            flat_message = torch.nanmean(metrics["message"], dim=0).flatten().numpy()
            correlation_rows.append({
                **base,
                "intervention": intervention,
                "rho_attention_response_total": spearman(flat_moved, flat_total),
                "rho_clean_attention_total": spearman(flat_clean, flat_total),
                "rho_route_message": spearman(flat_route, flat_message),
            })

        for intervention, metrics in estimation["output"].items():
            output_rows.append({
                **base,
                "intervention": intervention,
                "functional": _tensor_mean(metrics["functional"]),
                "beneficial": _tensor_mean(metrics["beneficial"]),
                "counterfactual_loss": _tensor_mean(metrics["counterfactual_loss"]),
                "counterfactual_accuracy": _tensor_mean(
                    metrics["counterfactual_accuracy"]
                ),
                "valid_graphs": int(torch.as_tensor(metrics["valid"]).sum()),
            })

        checks = estimation["checks"]
        check_rows.append({
            **base,
            "closure_absolute": float(checks["closure_absolute"]),
            "closure_relative": float(checks["closure_relative"]),
            "identical_noop": float(checks["noop_errors"].get("identical", float("nan"))),
            "record_permutation_noop": float(
                checks["noop_errors"].get("record_permutation", float("nan"))
            ),
            "structural_automorphism_noop": float(
                checks["noop_errors"].get(
                    "structural_record_record_noop", float("nan")
                )
            ),
            "onehop_layer2_address_raw_attention_moved_max": float(
                checks.get(
                    "onehop_layer2_address_raw_attention_moved_max", float("nan")
                )
            ),
            "onehop_layer2_address_gate_moved_max": float(
                checks.get("onehop_layer2_address_gate_moved_max", float("nan"))
            ),
            "onehop_layer2_address_profile_moved_max": float(
                checks.get("onehop_layer2_address_profile_moved_max", float("nan"))
            ),
            "onehop_layer2_address_profile_zero_pass": checks.get(
                "onehop_layer2_address_profile_zero_pass", ""
            ),
        })

        semantic = torch.nanmean(
            estimation["head"]["target_payload"]["total"], dim=0
        )
        structural = torch.nanmean(
            estimation["head"][STRUCTURAL_INTERVENTION]["total"], dim=0
        )
        semantic_norm = semantic / semantic.mean().clamp_min(EPS)
        structural_norm = structural / structural.mean().clamp_min(EPS)
        joint = 0.5 * (semantic_norm + structural_norm)
        selectivity = 0.5 * (semantic_norm - structural_norm)
        for layer in range(int(semantic.size(0))):
            for head in range(int(semantic.size(1))):
                context_rows.append({
                    **base,
                    "layer": layer,
                    "head": head,
                    "S_sem": float(semantic[layer, head]),
                    "S_str": float(structural[layer, head]),
                    "S_sem_norm": float(semantic_norm[layer, head]),
                    "S_str_norm": float(structural_norm[layer, head]),
                    "D": float(selectivity[layer, head]),
                    "J": float(joint[layer, head]),
                })

    grouped_cells: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in head_rows:
        grouped_cells[(
            row["model"], row["width"], row["N"], row["seed"],
            row["sample_type"], row["selected"], row["heldout_accuracy"],
            row["intervention"], row["layer"],
        )].append(row)
    cell_rows: list[dict[str, Any]] = []
    for key, values in grouped_cells.items():
        (
            model, width, records, seed, sample_type, selected,
            heldout_accuracy, intervention, layer,
        ) = key
        route = sum(float(row["route"]) for row in values)
        message = sum(float(row["message"]) for row in values)
        cell_rows.append({
            "model": model,
            "width": width,
            "N": records,
            "seed": seed,
            "sample_type": sample_type,
            "selected": selected,
            "heldout_accuracy": heldout_accuracy,
            "intervention": intervention,
            "layer": layer,
            "route_mass": route,
            "message_mass": message,
            "routing_share": route / max(route + message, EPS),
            "message_share": message / max(route + message, EPS),
            "retrieval_route": sum(float(row["retrieval_route"]) for row in values),
            "retrieval_attention_gate_moved": float(np.mean([
                float(row["retrieval_attention_gate_moved"]) for row in values
            ])),
            "retrieval_attention_profile_moved": float(np.mean([
                float(row["retrieval_attention_profile_moved"]) for row in values
            ])),
            "total": sum(float(row["total"]) for row in values),
        })
    cell_rows.sort(key=lambda row: (
        MODEL_ORDER.index(str(row["model"])), int(row["N"]), int(row["seed"]),
        str(row["intervention"]), int(row["layer"]),
    ))

    table_dir = analysis_root(cfg) / "tables"
    write_csv(table_dir / "head_metrics.csv", head_rows)
    write_csv(table_dir / "output_metrics.csv", output_rows)
    write_csv(table_dir / "head_correlations.csv", correlation_rows)
    write_csv(table_dir / "verification_checks.csv", check_rows)
    write_csv(table_dir / "specialisation_context.csv", context_rows)
    write_csv(table_dir / "mechanism_cells.csv", cell_rows)
    return {
        "heads": head_rows,
        "outputs": output_rows,
        "correlations": correlation_rows,
        "checks": check_rows,
        "context": context_rows,
        "cells": cell_rows,
    }


def build_causal_tables(
    caches: Sequence[Mapping[str, Any]],
    cfg: AnalysisConfig,
) -> dict[str, list[dict[str, Any]]]:
    ablation_rows: list[dict[str, Any]] = []
    patch_rows: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
    for cached in caches:
        checkpoint = cached["checkpoint"]
        base = {
            "model": checkpoint["model"],
            "width": int(checkpoint["width"]),
            "N": int(checkpoint["N"]),
            "seed": int(checkpoint["seed"]),
            "heldout_accuracy": float(checkpoint["heldout_accuracy"]),
        }
        ablation_rows.extend({**base, **row} for row in cached["ablation_rows"])
        patch_rows.extend({**base, **row} for row in cached["patch_rows"])
        families = cached["families"]
        family_rows.extend([
            {**base, "family": "routing", "heads": str(families["routing"])},
            {**base, "family": "message", "heads": str(families["message"])},
            {**base, "family": "overlap", "heads": str(families["overlap"])},
        ])
    table_dir = analysis_root(cfg) / "tables"
    write_csv(table_dir / "causal_ablation.csv", ablation_rows)
    write_csv(table_dir / "causal_patching.csv", patch_rows)
    write_csv(table_dir / "causal_families.csv", family_rows)
    return {"ablation": ablation_rows, "patching": patch_rows, "families": family_rows}


def configure_plots() -> None:
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "font.size": 10.5,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "legend.frameon": False,
    })


def save_figure(fig: Any, cfg: AnalysisConfig, stem: str) -> None:
    directory = analysis_root(cfg) / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{stem}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(directory / f"{stem}.pdf", bbox_inches="tight", facecolor="white")


def _panel(axis: Any, letter: str, title: str) -> None:
    axis.set_title(f"{letter}  {title}", loc="left")


def plot_capacity(
    manifest: Sequence[Mapping[str, Any]], cfg: AnalysisConfig
) -> None:
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(7.2, 5.0))
    for model in cfg.models:
        means, errors = [], []
        for records in cfg.ns:
            rows = [
                row for row in manifest
                if row["model"] == model and int(row["N"]) == records
            ]
            values = [float(row["heldout_accuracy"]) for row in rows]
            mean, error = mean_ci(values)
            means.append(mean)
            errors.append(error)
            axis.scatter(
                [records] * len(values), values,
                color=MODEL_COLOURS[model], alpha=0.25, s=28, zorder=2,
            )
            selected = next(row for row in rows if bool(row["selected_for_analysis"]))
            axis.scatter(
                [records], [selected["heldout_accuracy"]],
                facecolors="none", edgecolors=MODEL_COLOURS[model], s=80, lw=1.4,
                zorder=4,
            )
        axis.errorbar(
            cfg.ns, means, yerr=errors,
            color=MODEL_COLOURS[model], marker=MODEL_MARKERS[model],
            lw=2.1, capsize=3, label=model.replace("hop", "-hop"), zorder=3,
        )
    axis.plot(cfg.ns, [1.0 / value for value in cfg.ns], ":", color="#666666", label="chance")
    axis.set_xscale("log", base=2)
    axis.set_xticks(cfg.ns, [str(value) for value in cfg.ns])
    axis.set_ylim(-0.03, 1.08)
    axis.set_xlabel("Memory size N")
    axis.set_ylabel("Held-out recall accuracy")
    _panel(axis, "A", f"NAR capacity (width {cfg.analysis_width})")
    axis.legend(ncol=2)
    fig.tight_layout()
    save_figure(fig, cfg, "01_capacity")
    plt.close(fig)


def _primary(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [row for row in rows if row.get("sample_type") == "primary"]


def plot_specialisation_context(
    rows: Sequence[Mapping[str, Any]], cfg: AnalysisConfig
) -> None:
    import matplotlib.pyplot as plt

    selected = _primary(rows)
    fig, axes = plt.subplots(2, len(cfg.models), figsize=(5.1 * len(cfg.models), 8.4))
    axes = np.asarray(axes).reshape(2, len(cfg.models))
    for column, model in enumerate(cfg.models):
        model_rows = [row for row in selected if row["model"] == model]
        for layer in (0, 1):
            layer_rows = [row for row in model_rows if int(row["layer"]) == layer]
            for records in cfg.ns:
                points = [row for row in layer_rows if int(row["N"]) == records]
                axes[0, column].scatter(
                    [row["S_str"] for row in points],
                    [row["S_sem"] for row in points],
                    color=LAYER_COLOURS[layer], marker=N_MARKERS.get(records, "o"),
                    s=36, alpha=0.72,
                    label=f"L{layer + 1}" if records == cfg.ns[0] else None,
                )
                axes[1, column].scatter(
                    [row["D"] for row in points],
                    [row["J"] for row in points],
                    color=LAYER_COLOURS[layer], marker=N_MARKERS.get(records, "o"),
                    s=36, alpha=0.72,
                )
        axes[0, column].set_xlabel("Structural score $S_{str}$")
        axes[0, column].set_ylabel("Semantic score $S_{sem}$" if column == 0 else "")
        axes[1, column].set_xlabel("Selectivity $D$")
        axes[1, column].set_ylabel("Joint strength $J$" if column == 0 else "")
        axes[1, column].axvline(0, color="#777777", lw=1)
        _panel(axes[0, column], chr(ord("A") + column), model.replace("hop", "-hop"))
        _panel(axes[1, column], chr(ord("D") + column), "D-J context")
    axes[0, 0].legend()
    fig.suptitle(
        "Exploratory semantic-structural head allocation in NAR\n"
        "Colour denotes layer; marker denotes N. NAR has no independent structural target.",
        y=1.01,
    )
    fig.tight_layout()
    save_figure(fig, cfg, "02_specialisation_context")
    plt.close(fig)


def _scatter_metric_grid(
    rows: Sequence[Mapping[str, Any]],
    cfg: AnalysisConfig,
    *,
    x_key: str,
    y_key: str,
    stem: str,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    import matplotlib.pyplot as plt

    interventions = ("target_payload", "address_different_answer")
    selected = _primary(rows)
    fig, axes = plt.subplots(2, len(cfg.models), figsize=(5.0 * len(cfg.models), 8.0))
    axes = np.asarray(axes).reshape(2, len(cfg.models))
    for row_index, intervention in enumerate(interventions):
        for column, model in enumerate(cfg.models):
            axis = axes[row_index, column]
            points = [
                row for row in selected
                if row["model"] == model and row["intervention"] == intervention
            ]
            for layer in (0, 1):
                layer_points = [row for row in points if int(row["layer"]) == layer]
                for records in cfg.ns:
                    cell = [row for row in layer_points if int(row["N"]) == records]
                    axis.scatter(
                        [row[x_key] for row in cell], [row[y_key] for row in cell],
                        color=LAYER_COLOURS[layer], marker=N_MARKERS.get(records, "o"),
                        s=34, alpha=0.68,
                    )
            rho = spearman(
                [float(row[x_key]) for row in points],
                [float(row[y_key]) for row in points],
            )
            axis.text(0.04, 0.95, rf"descriptive $\rho={rho:.2f}$", transform=axis.transAxes, va="top")
            axis.set_xlabel(xlabel)
            if column == 0:
                axis.set_ylabel(ylabel)
            label = "payload" if row_index == 0 else "address"
            _panel(axis, chr(ord("A") + row_index * len(cfg.models) + column), f"{model}: {label}")
    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    save_figure(fig, cfg, stem)
    plt.close(fig)


def plot_attention_and_decomposition(
    rows: Sequence[Mapping[str, Any]], cfg: AnalysisConfig
) -> None:
    _scatter_metric_grid(
        rows, cfg,
        x_key="attention_ratio", y_key="total",
        stem="03a_clean_attention_vs_transport",
        title="Clean queried-record attention versus output-grounded transport",
        xlabel="Target/background attention ratio", ylabel="Transport score",
    )
    _scatter_metric_grid(
        rows, cfg,
        x_key="attention_moved_normalized", y_key="total",
        stem="03b_attention_response_vs_transport",
        title="Intervention-induced attention movement versus transport",
        xlabel="Moved attention mass per valid edge", ylabel="Transport score",
    )
    _scatter_metric_grid(
        rows, cfg,
        x_key="message", y_key="route",
        stem="03c_routing_message_decomposition",
        title="Output-grounded routing-message decomposition",
        xlabel="Message contribution", ylabel="Routing contribution",
    )


def _cell_head_mass(
    rows: Sequence[Mapping[str, Any]],
    *,
    model: str,
    records: int,
    intervention: str,
    layer: int,
) -> dict[str, float]:
    cell = [
        row for row in rows
        if row["model"] == model
        and int(row["N"]) == records
        and row["intervention"] == intervention
        and int(row["layer"]) == layer
    ]
    route = sum(float(row["route"]) for row in cell)
    message = sum(float(row["message"]) for row in cell)
    return {
        "route": route,
        "message": message,
        "routing_share": route / max(route + message, EPS),
        "retrieval_route": sum(float(row["retrieval_route"]) for row in cell),
        "retrieval_attention_gate_moved": float(np.mean([
            float(row["retrieval_attention_gate_moved"]) for row in cell
        ])),
        "retrieval_attention_profile_moved": float(np.mean([
            float(row["retrieval_attention_profile_moved"]) for row in cell
        ])),
    }


def plot_information_rendezvous(
    head_rows: Sequence[Mapping[str, Any]],
    cell_rows: Sequence[Mapping[str, Any]],
    manifest: Sequence[Mapping[str, Any]],
    cfg: AnalysisConfig,
) -> None:
    import matplotlib.pyplot as plt

    rows = _primary(head_rows)
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.2))
    axes = np.asarray(axes).reshape(-1)
    for model in cfg.models:
        address_share = [
            _cell_head_mass(
                rows, model=model, records=records,
                intervention="address_different_answer", layer=1,
            )["routing_share"]
            for records in cfg.ns
        ]
        axes[0].plot(
            cfg.ns, address_share, color=MODEL_COLOURS[model],
            marker=MODEL_MARKERS[model], lw=2, label=model,
        )
        for records in cfg.anchor_ns:
            seed_values = [
                float(row["routing_share"])
                for row in cell_rows
                if row["model"] == model
                and int(row["N"]) == records
                and row["intervention"] == "address_different_answer"
                and int(row["layer"]) == 1
            ]
            mean, error = mean_ci(seed_values)
            axes[0].errorbar(
                [records], [mean], yerr=[error], color=MODEL_COLOURS[model],
                marker=MODEL_MARKERS[model], markerfacecolor="white",
                capsize=3, lw=1.2, zorder=5,
            )
        for layer, linestyle in ((0, "--"), (1, "-")):
            values = [
                1.0 - _cell_head_mass(
                    rows, model=model, records=records,
                    intervention="target_payload", layer=layer,
                )["routing_share"]
                for records in cfg.ns
            ]
            axes[1].plot(
                cfg.ns, values, color=MODEL_COLOURS[model], linestyle=linestyle,
                marker=MODEL_MARKERS[model], lw=1.8,
                label=f"{model} L{layer + 1}",
            )
            for records in cfg.anchor_ns:
                seed_values = [
                    float(row["message_share"])
                    for row in cell_rows
                    if row["model"] == model
                    and int(row["N"]) == records
                    and row["intervention"] == "target_payload"
                    and int(row["layer"]) == layer
                ]
                mean, error = mean_ci(seed_values)
                axes[1].errorbar(
                    [records], [mean], yerr=[error], color=MODEL_COLOURS[model],
                    marker=MODEL_MARKERS[model], markerfacecolor="white",
                    capsize=2.5, lw=1.0, alpha=0.85,
                )
        for records in cfg.ns:
            profile = _cell_head_mass(
                rows, model=model, records=records,
                intervention="address_different_answer", layer=1,
            )["retrieval_attention_profile_moved"]
            retrieval = _cell_head_mass(
                rows, model=model, records=records,
                intervention="address_different_answer", layer=1,
            )["retrieval_route"]
            selected_checkpoint = next(
                row for row in manifest
                if row["model"] == model
                and int(row["N"]) == records
                and bool(row["selected_for_analysis"])
            )
            axes[2].scatter(
                records, profile,
                color=MODEL_COLOURS[model], marker=N_MARKERS.get(records, "o"),
                s=62,
            )
            axes[3].scatter(
                retrieval, selected_checkpoint["heldout_accuracy"],
                color=MODEL_COLOURS[model], marker=N_MARKERS.get(records, "o"),
                s=62,
            )
        for cell in cell_rows:
            if (
                cell["model"] == model
                and int(cell["N"]) in cfg.anchor_ns
                and cell["intervention"] == "address_different_answer"
                and int(cell["layer"]) == 1
            ):
                axes[3].scatter(
                    cell["retrieval_route"], cell["heldout_accuracy"],
                    color=MODEL_COLOURS[model], s=22, alpha=0.22,
                )
        axes[2].plot(
            cfg.ns,
            [
                _cell_head_mass(
                    rows, model=model, records=records,
                    intervention="address_different_answer", layer=1,
                )["retrieval_attention_profile_moved"]
                for records in cfg.ns
            ],
            color=MODEL_COLOURS[model], marker=MODEL_MARKERS[model],
            lw=2, label=model,
        )
        for records in cfg.anchor_ns:
            seed_values = [
                float(cell["retrieval_attention_profile_moved"])
                for cell in cell_rows
                if cell["model"] == model
                and int(cell["N"]) == records
                and cell["intervention"] == "address_different_answer"
                and int(cell["layer"]) == 1
            ]
            mean, error = mean_ci(seed_values)
            axes[2].errorbar(
                [records], [mean], yerr=[error], color=MODEL_COLOURS[model],
                marker=MODEL_MARKERS[model], markerfacecolor="white",
                capsize=3, lw=1.2, zorder=5,
            )
    for axis in axes[:3]:
        axis.set_xscale("log", base=2)
        axis.set_xticks(cfg.ns, [str(value) for value in cfg.ns])
        axis.set_xlabel("Memory size N")
    for axis in axes[:2]:
        axis.set_ylim(-0.04, 1.04)
    axes[0].set_ylabel("Layer-2 routing share")
    axes[1].set_ylabel("Payload message share")
    axes[2].set_ylabel("Within-record routing-profile movement")
    axes[3].set_xlabel("Layer-2 centre-to-record address routing")
    axes[3].set_ylabel("Held-out accuracy")
    _panel(axes[0], "A", "Address routing emerges at the rendezvous")
    _panel(axes[1], "B", "Payload carriage across depth")
    _panel(axes[2], "C", "Address-selective routing requires graph support")
    _panel(axes[3], "D", "Realised retrieval mechanism and capacity")
    axes[0].legend()
    axes[1].legend(fontsize=8, ncol=2)
    axes[2].legend()
    fig.tight_layout()
    save_figure(fig, cfg, "04_information_rendezvous")
    plt.close(fig)


def _group_mean(
    rows: Sequence[Mapping[str, Any]], key: str, **filters: Any
) -> float:
    values = [
        float(row[key]) for row in rows
        if all(row.get(name) == value for name, value in filters.items())
    ]
    values = [value for value in values if math.isfinite(value)]
    return float(np.mean(values)) if values else float("nan")


def plot_causal_validation(
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
    cfg: AnalysisConfig,
) -> None:
    import matplotlib.pyplot as plt

    ablation = tables["ablation"]
    patching = tables["patching"]
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.4))
    families = ("routing", "message", "routing_random", "message_random")
    colours = ("#9e0142", "#2c7bb6", "#d9d9d9", "#969696")
    ablation_means = [_group_mean(ablation, "loss_delta", family=family) for family in families]
    axes[0, 0].bar(range(len(families)), ablation_means, color=colours)
    axes[0, 0].set_xticks(range(len(families)), ["route", "message", "route null", "msg null"], rotation=15)
    axes[0, 0].set_ylabel("Cross-entropy increase")
    axes[0, 0].axhline(0, color="#777777", lw=1)
    _panel(axes[0, 0], "A", "Selected-family necessity")

    interventions = ("address_different_answer", "target_payload")
    short = ("address", "payload")
    for column, (metric, title) in enumerate((
        ("routing_rescue", "Routing-only rescue"),
        ("message_rescue", "Message-only rescue"),
    )):
        axis = axes[0, 1] if column == 0 else axes[1, 0]
        width = 0.34
        for family_index, family in enumerate(("routing", "message")):
            values = [
                _group_mean(patching, metric, family=family, intervention=intervention)
                for intervention in interventions
            ]
            positions = np.arange(2) + (family_index - 0.5) * width
            axis.bar(
                positions, values, width=width,
                color=MODEL_COLOURS["dense"] if family == "routing" else MODEL_COLOURS["2hop"],
                alpha=0.82, label=f"{family} heads",
            )
        axis.set_xticks(range(2), short)
        axis.set_ylabel("Clean-target log-probability rescue")
        axis.axhline(0, color="#777777", lw=1)
        _panel(axis, "B" if column == 0 else "C", title)
        axis.legend()

    interaction_values = [
        _group_mean(patching, "interaction", intervention=intervention, family=family)
        for intervention in interventions
        for family in ("routing", "message")
    ]
    axes[1, 1].bar(
        range(4), interaction_values,
        color=["#9e0142", "#2c7bb6", "#9e0142", "#2c7bb6"], alpha=0.82,
    )
    axes[1, 1].set_xticks(
        range(4), ["addr-route", "addr-msg", "pay-route", "pay-msg"], rotation=18
    )
    axes[1, 1].axhline(0, color="#777777", lw=1)
    axes[1, 1].set_ylabel("Finite downstream interaction")
    _panel(axes[1, 1], "D", "Routing-message interaction after wV")
    fig.suptitle(
        "Targeted causal validation of routing and message specialisation",
        y=1.01,
    )
    fig.tight_layout()
    save_figure(fig, cfg, "05_targeted_causal_validation")
    plt.close(fig)


def build_summary(
    manifest: Sequence[Mapping[str, Any]],
    metric_tables: Mapping[str, Sequence[Mapping[str, Any]]],
    causal_tables: Mapping[str, Sequence[Mapping[str, Any]]],
    cfg: AnalysisConfig,
) -> dict[str, Any]:
    heads = _primary(metric_tables["heads"])
    checks = metric_tables["checks"]
    onehop_profile_zero = max(
        [
            float(row["onehop_layer2_address_profile_moved_max"])
            for row in checks
            if row["model"] == "1hop"
            and math.isfinite(
                float(row["onehop_layer2_address_profile_moved_max"])
            )
        ]
        or [float("nan")]
    )
    payload_cells = [
        _cell_head_mass(
            heads, model=model, records=records,
            intervention="target_payload", layer=layer,
        )
        for model in cfg.models for records in cfg.ns for layer in (0, 1)
    ]
    payload_message_share = float(np.mean([
        1.0 - item["routing_share"] for item in payload_cells
    ]))
    nonlocal_values = [
        _cell_head_mass(
            heads, model=model, records=records,
            intervention="address_different_answer", layer=1,
        )["routing_share"]
        for model in cfg.models if model != "1hop" for records in cfg.ns
    ]
    address_route_dense = (
        float(np.mean(nonlocal_values)) if nonlocal_values else float("nan")
    )
    onehop_values = [
        _cell_head_mass(
            heads, model="1hop", records=records,
            intervention="address_different_answer", layer=1,
        )["routing_share"]
        for records in cfg.ns
    ] if "1hop" in cfg.models else []
    address_route_onehop = (
        float(np.mean(onehop_values)) if onehop_values else float("nan")
    )
    patching = causal_tables["patching"]
    routing_double = (
        _group_mean(
            patching, "routing_rescue", family="routing",
            intervention="address_different_answer",
        )
        - _group_mean(
            patching, "message_rescue", family="routing",
            intervention="address_different_answer",
        )
    )
    message_double = (
        _group_mean(
            patching, "message_rescue", family="message",
            intervention="target_payload",
        )
        - _group_mean(
            patching, "routing_rescue", family="message",
            intervention="target_payload",
        )
    )
    hypotheses = {
        "H1_onehop_address_selective_routing_zero": {
            "max_within_record_profile_movement": onehop_profile_zero,
            "status": "supported"
            if onehop_profile_zero <= cfg.noop_tol else "unsupported",
        },
        "H2_payload_message_dominance": {
            "mean_message_share": payload_message_share,
            "status": "supported" if payload_message_share > 0.5 else "unsupported",
        },
        "H3_nonlocal_address_routing_exceeds_onehop": {
            "nonlocal": address_route_dense,
            "onehop": address_route_onehop,
            "status": "supported" if address_route_dense > address_route_onehop else "unsupported",
        },
        "H5_causal_double_dissociation": {
            "address_routing_advantage": routing_double,
            "payload_message_advantage": message_double,
            "status": "supported" if routing_double > 0 and message_double > 0 else "unsupported",
        },
    }
    return {
        "analysis_version": ANALYSIS_VERSION,
        "fingerprint": scientific_fingerprint(cfg),
        "width": cfg.analysis_width,
        "config": asdict(cfg),
        "checkpoint_cells": len(manifest),
        "hypotheses": hypotheses,
        "artifacts": {
            "tables": sorted(path.name for path in (analysis_root(cfg) / "tables").glob("*.csv")),
            "figures": sorted(path.name for path in (analysis_root(cfg) / "figures").glob("*.png")),
        },
    }


def make_figures_and_tables(
    manifest: Sequence[Mapping[str, Any]],
    metric_caches: Sequence[Mapping[str, Any]],
    causal_caches: Sequence[Mapping[str, Any]],
    cfg: AnalysisConfig,
) -> dict[str, Any]:
    configure_plots()
    metric_tables = build_metric_tables(metric_caches, cfg)
    causal_tables = build_causal_tables(causal_caches, cfg)
    plot_capacity(manifest, cfg)
    plot_specialisation_context(metric_tables["context"], cfg)
    plot_attention_and_decomposition(metric_tables["heads"], cfg)
    plot_information_rendezvous(
        metric_tables["heads"], metric_tables["cells"], manifest, cfg
    )
    plot_causal_validation(causal_tables, cfg)
    summary = build_summary(manifest, metric_tables, causal_tables, cfg)
    atomic_write_json(analysis_root(cfg) / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive-root", default=nar.DEFAULT_DRIVE_ROOT)
    parser.add_argument("--run-name", default="nar_grit_fixed_n_v3")
    parser.add_argument("--analysis-width", type=int, choices=(64, 128), default=64)
    parser.add_argument("--phase", choices=("all", "index", "analyze", "causal", "figures"), default="all")
    parser.add_argument("--models", default=",".join(MODEL_ORDER))
    parser.add_argument("--ns", default="4,8,16,32,64")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--anchor-ns", default="4,16,64")
    parser.add_argument("--donors", type=int, default=4)
    parser.add_argument("--discovery-graphs", type=int, default=32)
    parser.add_argument("--mechanism-graphs", type=int, default=96)
    parser.add_argument("--robustness-graphs", type=int, default=48)
    parser.add_argument("--causal-graphs", type=int, default=256)
    parser.add_argument("--causal-donors", type=int, default=4)
    parser.add_argument("--family-size", type=int, default=2)
    parser.add_argument("--random-families", type=int, default=8)
    parser.add_argument("--max-batch-nodes", type=int, default=4500)
    parser.add_argument("--max-replica-pairs", type=int, default=1_200_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--grit-dir", default=nar.DEFAULT_GRIT_DIR)
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--force-index", action="store_true")
    parser.add_argument("--force-analysis", action="store_true")
    parser.add_argument("--force-causal", action="store_true")
    parser.add_argument("--fast-dev-run", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> AnalysisConfig:
    values: dict[str, Any] = {
        "drive_root": args.drive_root,
        "run_name": args.run_name,
        "analysis_width": args.analysis_width,
        "models": parse_str_tuple(args.models),
        "ns": parse_int_tuple(args.ns),
        "seeds": parse_int_tuple(args.seeds),
        "anchor_ns": parse_int_tuple(args.anchor_ns),
        "donors": args.donors,
        "discovery_graphs": args.discovery_graphs,
        "mechanism_graphs": args.mechanism_graphs,
        "robustness_graphs": args.robustness_graphs,
        "causal_graphs": args.causal_graphs,
        "causal_donors": args.causal_donors,
        "family_size": args.family_size,
        "random_families": args.random_families,
        "max_batch_nodes": args.max_batch_nodes,
        "max_replica_pairs": args.max_replica_pairs,
        "device": args.device,
        "grit_dir": args.grit_dir,
    }
    if args.fast_dev_run:
        values.update({
            "models": ("1hop", "dense"),
            "ns": (4,),
            "anchor_ns": (4,),
            "seeds": (0,),
            "donors": 1,
            "discovery_graphs": 2,
            "mechanism_graphs": 2,
            "robustness_graphs": 2,
            "causal_graphs": 2,
            "causal_donors": 1,
            "family_size": 1,
            "random_families": 2,
            "max_batch_nodes": 256,
            "max_replica_pairs": 20_000,
        })
    cfg = AnalysisConfig(**values)
    cfg.validate()
    return cfg


def main(argv: Sequence[str] | None = None) -> dict[str, Any] | None:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    cfg = config_from_args(args)
    root = analysis_root(cfg)
    root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        root / "config.json",
        {
            "analysis_version": ANALYSIS_VERSION,
            "fingerprint": scientific_fingerprint(cfg),
            "config": asdict(cfg),
            "created": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        },
    )
    manifest = build_checkpoint_manifest(cfg, force=args.force_index)
    schedule = analysis_schedule(manifest, cfg)
    causal_rows = causal_schedule(manifest, cfg)
    if args.phase == "index":
        print(f"[done] checkpoint manifest: {root / 'checkpoint_manifest.csv'}", flush=True)
        return None

    if args.phase in ("all", "analyze", "causal"):
        nar.setup_official_grit(Path(cfg.grit_dir), install=not args.skip_install)
        device = resolve_device(cfg.device)
        atomic_write_json(
            root / "environment.json",
            {
                "python": os.sys.version,
                "torch": torch.__version__,
                "device": str(device),
                "cuda": torch.version.cuda,
                "official_grit_commit": nar.OFFICIAL_GRIT_COMMIT,
                "time": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
            },
        )
    else:
        device = torch.device("cpu")

    if args.phase in ("all", "analyze"):
        for row in schedule:
            analyze_checkpoint(
                row,
                cfg,
                device=device,
                force=args.force_analysis,
            )
        if args.phase == "analyze":
            print("[done] metric caches complete; run --phase causal", flush=True)
            return None

    metric_caches = load_metric_caches(cfg, schedule)
    metric_by_cell = {
        (
            cache["checkpoint"]["model"],
            int(cache["checkpoint"]["N"]),
            int(cache["checkpoint"]["seed"]),
        ): cache
        for cache in metric_caches
    }

    if args.phase in ("all", "causal"):
        for row in causal_rows:
            key = (str(row["model"]), int(row["N"]), int(row["seed"]))
            run_causal_checkpoint(
                row,
                metric_by_cell[key],
                cfg,
                device=device,
                force=args.force_causal,
            )
        if args.phase == "causal":
            print("[done] causal caches complete; run --phase figures", flush=True)
            return None

    causal_caches = load_causal_caches(cfg, causal_rows)
    summary = make_figures_and_tables(
        manifest,
        metric_caches,
        causal_caches,
        cfg,
    )
    print(f"[done] NAR transport mechanisms saved under {root}", flush=True)
    return summary


if __name__ == "__main__":
    main()
