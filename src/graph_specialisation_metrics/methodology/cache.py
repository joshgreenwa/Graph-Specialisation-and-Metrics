"""Resumable canonical caches and fail-closed paper-artifact metadata."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np

from ..carriage.env import log
from .protocol import PROTOCOL_VERSION, stable_hash


class StaleCacheError(RuntimeError):
    pass


def _scientific_contract_record(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Return validity fields while retaining checkout identity only as provenance."""

    record = dict(contract)
    record.pop("repository_commit", None)
    return record


def _scientific_contract_fingerprint(contract: Mapping[str, Any]) -> str:
    return stable_hash(_scientific_contract_record(contract))


def _contract_difference_paths(
    stored: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    prefix: str = "",
) -> list[str]:
    differences: list[str] = []
    for key in sorted(set(stored) | set(expected)):
        path = f"{prefix}.{key}" if prefix else str(key)
        if key not in stored or key not in expected:
            differences.append(path)
            continue
        left = stored[key]
        right = expected[key]
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            differences.extend(
                _contract_difference_paths(left, right, prefix=path)
            )
        elif left != right:
            differences.append(path)
    return differences


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
        # The source commit remains in stored metadata for provenance, but changing an unrelated
        # checkout commit must not invalidate an otherwise identical numerical artifact.
        return _scientific_contract_fingerprint(dataclasses.asdict(self))


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

    Direct users are fail-closed by default. Resumable runners may opt into ``archive`` handling:
    an incompatible derived cache is moved into the local ``_stale`` archive and becomes a cache
    miss, preserving the original bytes while allowing the requested contract to be recomputed.
    """

    def __init__(
        self,
        root: str | Path,
        contract: CacheContract,
        *,
        stale_policy: Literal["raise", "archive"] = "raise",
    ):
        if stale_policy not in {"raise", "archive"}:
            raise ValueError("stale_policy must be 'raise' or 'archive'")
        self.root = Path(root)
        self.contract = contract
        self.stale_policy = stale_policy

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
                # Same-contract forced recomputation may replace the cache. In archive mode, an
                # incompatible derived cache is preserved under _stale before this write.
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
                "provenance_fingerprint": stable_hash(
                    dataclasses.asdict(self.contract)
                ),
            },
            "value": value,
        }
        temporary = path.with_suffix(path.suffix + ".partial")
        torch.save(payload, temporary)
        temporary.replace(path)
        return path

    def _archive_stale(
        self,
        path: Path,
        *,
        reason: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        cache_root = (
            self.root
            / self.contract.task
            / f"seed_{self.contract.train_seed}"
            / "cache"
        )
        try:
            relative = path.relative_to(cache_root)
        except ValueError:
            relative = Path(path.name)
        claimed = str((metadata or {}).get("contract_fingerprint", "unknown"))
        token = f"{claimed[:12]}-{uuid.uuid4().hex[:8]}"
        archive = (
            cache_root
            / "_stale"
            / relative.parent
            / f"{relative.stem}.stale-{token}{relative.suffix}"
        )
        archive.parent.mkdir(parents=True, exist_ok=True)
        path.replace(archive)
        try:
            atomic_json(
                archive.with_suffix(archive.suffix + ".json"),
                {
                    "reason": reason,
                    "original_path": str(path),
                    "archived_path": str(archive),
                    "expected_contract": dataclasses.asdict(self.contract),
                    "stored_metadata": dict(metadata or {}),
                },
            )
        except OSError:
            # The numerical artifact is already safely archived; a sidecar failure should not
            # turn a recoverable cache miss back into a failed production run.
            pass
        log(
            f"[cache] archived incompatible derived cache {path} -> {archive}; "
            f"recomputing ({reason})"
        )
        return archive

    def _stale(
        self,
        path: Path,
        message: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if self.stale_policy == "raise":
            raise StaleCacheError(message)
        self._archive_stale(path, reason=message, metadata=metadata)

    def load(self, stage: str, name: str, *, strict: bool = False) -> Any | None:
        import torch

        del strict  # Kept for API compatibility; stale behavior is selected at construction.
        path = self.path(stage, name)
        if not path.exists():
            return None
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except (OSError, RuntimeError, EOFError) as error:
            message = f"cache {path} is unreadable: {type(error).__name__}"
            if self.stale_policy == "raise":
                raise StaleCacheError(message) from error
            self._stale(path, message)
            return None
        if not isinstance(payload, Mapping):
            self._stale(path, f"cache {path} is malformed")
            return None
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, Mapping):
            self._stale(path, f"cache {path} has malformed metadata")
            return None
        expected = self.contract.fingerprint
        if metadata.get("protocol_version") != PROTOCOL_VERSION:
            self._stale(
                path,
                f"cache {path} uses protocol "
                f"{metadata.get('protocol_version')!r}; expected {PROTOCOL_VERSION!r}",
                metadata=metadata,
            )
            return None
        stored_contract = metadata.get("contract")
        if not isinstance(stored_contract, Mapping):
            self._stale(path, f"cache {path} has a malformed contract", metadata=metadata)
            return None
        claimed = metadata.get("contract_fingerprint")
        stored_scientific = _scientific_contract_fingerprint(stored_contract)
        # v4 caches written before repository commits became provenance-only used the complete
        # record for this claim. Accept that legacy self-consistent representation.
        legacy_claim = stable_hash(dict(stored_contract))
        if claimed not in {stored_scientific, legacy_claim}:
            self._stale(
                path,
                f"cache {path} has an internally inconsistent contract fingerprint",
                metadata=metadata,
            )
            return None
        provenance_claim = metadata.get("provenance_fingerprint")
        if provenance_claim is not None and provenance_claim != legacy_claim:
            self._stale(
                path,
                f"cache {path} has an internally inconsistent provenance fingerprint",
                metadata=metadata,
            )
            return None
        if stored_scientific != expected:
            differences = _contract_difference_paths(
                _scientific_contract_record(stored_contract),
                _scientific_contract_record(dataclasses.asdict(self.contract)),
            )
            detail = ", ".join(differences[:8]) or "unknown"
            if len(differences) > 8:
                detail += f", +{len(differences) - 8} more"
            self._stale(
                path,
                f"cache {path} does not match this scientific contract; "
                f"differing fields: {detail}",
                metadata=metadata,
            )
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
    scientific = _scientific_contract_fingerprint(contract)
    legacy = stable_hash(dict(contract))
    if claimed not in {scientific, legacy}:
        raise StaleCacheError(
            f"cache contract fingerprint is internally inconsistent: {resolved}"
        )
    provenance_claim = metadata.get("provenance_fingerprint")
    if provenance_claim is not None and provenance_claim != legacy:
        raise StaleCacheError(
            f"cache provenance fingerprint is internally inconsistent: {resolved}"
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
