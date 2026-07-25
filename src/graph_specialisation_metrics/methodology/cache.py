"""Fail-closed canonical caches and paper-artifact metadata."""

from __future__ import annotations

import dataclasses
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..scoring_refinement.cache import sha256_file
from .protocol import PROTOCOL_VERSION, stable_hash


class StaleCacheError(RuntimeError):
    pass


@dataclass(frozen=True)
class CacheContract:
    protocol_fingerprint: str
    task: str
    task_adapter_version: str
    checkpoint_sha256: str
    train_seed: int
    model_geometry: Mapping[str, Any]
    output_representation: str
    sigma: tuple[float, ...]
    split_fingerprint: str
    event_manifest_hash: str
    donors_per_source: int
    source_cap: int
    bootstrap_seed: int
    bootstrap_replicates: int = 2_000
    raw_score_aggregation: str = "donor->source->graph"
    semantic_donor_law: str = "graph-uniform/node-uniform/min-gap/iid-replacement"
    structural_donor_law: str = "node-uniform/min-gap/iid-replacement"
    functional_estimand: str = "F_sens"
    beneficial_sign: str = "positive-is-beneficial"

    @property
    def fingerprint(self) -> str:
        return stable_hash(dataclasses.asdict(self))


def checkpoint_sha256(path: str | Path) -> str:
    return sha256_file(Path(path))


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
    except ImportError:
        pass
    raise TypeError(f"cannot JSON encode {type(value)!r}")


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=path.name, suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, default=_json_default)
            stream.write("\n")
        Path(temporary).replace(path)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()


class CanonicalCache:
    """Cache namespace keyed by the complete scientific contract."""

    def __init__(self, root: str | Path, contract: CacheContract):
        self.root = Path(root)
        self.contract = contract

    def path(self, stage: str, name: str, suffix: str = ".pt") -> Path:
        return (
            self.root
            / self.contract.task
            / f"seed_{self.contract.train_seed}"
            / "cache"
            / stage
            / f"{name}{suffix}"
        )

    def save(self, stage: str, name: str, value: Any) -> Path:
        import torch

        path = self.path(stage, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": {
                "protocol_version": PROTOCOL_VERSION,
                "contract": dataclasses.asdict(self.contract),
                "contract_fingerprint": self.contract.fingerprint,
            },
            "value": value,
        }
        temporary = path.with_suffix(path.suffix + ".partial")
        torch.save(payload, temporary)
        temporary.replace(path)
        return path

    def load(self, stage: str, name: str, *, strict: bool = False) -> Any | None:
        import torch

        path = self.path(stage, name)
        if not path.exists():
            return None
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except (OSError, RuntimeError, EOFError) as error:
            if strict:
                raise StaleCacheError(f"unreadable cache {path}") from error
            return None
        metadata = payload.get("metadata", {})
        expected = self.contract.fingerprint
        if metadata.get("protocol_version") != PROTOCOL_VERSION:
            if strict:
                raise StaleCacheError(f"{path} uses another methodology version")
            return None
        if metadata.get("contract_fingerprint") != expected:
            if strict:
                raise StaleCacheError(f"{path} does not match this scientific contract")
            return None
        return payload.get("value")

    def save_audit(self, name: str, payload: Any) -> Path:
        path = (
            self.root
            / self.contract.task
            / f"seed_{self.contract.train_seed}"
            / "audit"
            / f"{name}.json"
        )
        atomic_json(
            path,
            {
                "protocol_version": PROTOCOL_VERSION,
                "contract": dataclasses.asdict(self.contract),
                "payload": payload,
            },
        )
        return path

