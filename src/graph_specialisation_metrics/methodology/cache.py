"""Fail-closed canonical caches and paper-artifact metadata."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .protocol import PROTOCOL_VERSION, stable_hash


class StaleCacheError(RuntimeError):
    pass


@dataclass(frozen=True)
class ReadOnlyCacheArtifact:
    """A validated immutable cache payload with its original provenance."""

    path: Path
    file_sha256: str
    metadata: Mapping[str, Any]
    value: Any


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
    repository_commit: str = "unknown"
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
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    """Cache namespace keyed by the complete scientific contract.

    Existing cache files are immutable across contracts. A stale file is a protected scientific
    artifact, not a cache miss: callers must select a new output directory or analysis name rather
    than replacing it in place.
    """

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
        if path.exists():
            try:
                # A forced recomputation may replace a cache only when its complete scientific
                # contract is unchanged. Incompatible or unreadable files remain untouched.
                self.load(stage, name, strict=True)
            except StaleCacheError as error:
                raise StaleCacheError(
                    f"refusing to overwrite protected cache {path}; its stored contract does "
                    "not match the current run (or the file is unreadable). Choose a different "
                    "output_dir/analysis name. The existing file was left untouched."
                ) from error
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

        del strict  # Existing incompatible files are always protected, independent of audit mode.
        path = self.path(stage, name)
        if not path.exists():
            return None
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except (OSError, RuntimeError, EOFError) as error:
            raise StaleCacheError(
                f"protected cache {path} is unreadable and will not be replaced"
            ) from error
        if not isinstance(payload, Mapping):
            raise StaleCacheError(
                f"protected cache {path} is malformed and will not be replaced"
            )
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise StaleCacheError(
                f"protected cache {path} has malformed metadata and will not be replaced"
            )
        expected = self.contract.fingerprint
        if metadata.get("protocol_version") != PROTOCOL_VERSION:
            raise StaleCacheError(
                f"protected cache {path} uses another methodology version and will not "
                "be replaced; choose a different output_dir/analysis name"
            )
        if metadata.get("contract_fingerprint") != expected:
            raise StaleCacheError(
                f"protected cache {path} does not match this scientific contract and will not "
                "be replaced; choose a different output_dir/analysis name"
            )
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


def load_cache_artifact_file(path: str | Path) -> ReadOnlyCacheArtifact:
    """Load and validate an existing cache without comparing it to the current checkout.

    This is intentionally read-only. It verifies the cache's own protocol and stored contract
    fingerprint, then exposes that original contract so an additive analysis can bind itself to
    the exact artifact even when the repository has since advanced.
    """

    import torch

    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"cache file does not exist: {resolved}")
    try:
        payload = torch.load(resolved, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, EOFError) as error:
        raise StaleCacheError(f"unreadable cache {resolved}") from error
    if (
        not isinstance(payload, Mapping)
        or "metadata" not in payload
        or "value" not in payload
    ):
        raise StaleCacheError(f"cache payload is malformed: {resolved}")
    metadata = payload["metadata"]
    if metadata.get("protocol_version") != PROTOCOL_VERSION:
        raise StaleCacheError(
            f"{resolved} uses protocol {metadata.get('protocol_version')!r}; "
            f"expected {PROTOCOL_VERSION!r}"
        )
    contract = metadata.get("contract")
    if not isinstance(contract, Mapping):
        raise StaleCacheError(f"cache contract is malformed: {resolved}")
    claimed = metadata.get("contract_fingerprint")
    actual = stable_hash(dict(contract))
    if claimed != actual:
        raise StaleCacheError(
            f"cache contract fingerprint is internally inconsistent: {resolved}"
        )
    return ReadOnlyCacheArtifact(
        path=resolved,
        file_sha256=checkpoint_sha256(resolved),
        metadata=metadata,
        value=payload["value"],
    )


def load_cache_value_file(path: str | Path) -> Any:
    """Load a consolidated cache for a model-free figures-only pass."""

    return load_cache_artifact_file(path).value
