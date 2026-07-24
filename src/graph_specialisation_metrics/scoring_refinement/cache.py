"""Small atomic cache layer with explicit protocol/checkpoint fingerprints."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .config import stable_hash


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
    except ImportError:
        pass
    raise TypeError(f"cannot JSON encode {type(value)!r}")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=path.name, suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        Path(temporary).replace(path)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if str(key) not in fields:
                fields.append(str(key))
    fd, temporary = tempfile.mkstemp(prefix=path.name, suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            if fields:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(dict(row) for row in rows)
        Path(temporary).replace(path)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class CacheStore:
    """Namespaced cache whose items reject stale metadata."""

    def __init__(
        self,
        root: Path,
        *,
        protocol_fingerprint: str,
        checkpoint_sha: str,
        task: str,
    ) -> None:
        self.root = Path(root)
        self.protocol_fingerprint = str(protocol_fingerprint)
        self.checkpoint_sha = str(checkpoint_sha)
        self.task = str(task)

    def path(self, namespace: str, name: str, suffix: str = ".pt") -> Path:
        return self.root / "cache" / namespace / f"{name}{suffix}"

    @property
    def metadata(self) -> dict[str, str]:
        return {
            "protocol_fingerprint": self.protocol_fingerprint,
            "checkpoint_sha256": self.checkpoint_sha,
            "task": self.task,
        }

    def save_torch(
        self,
        namespace: str,
        name: str,
        value: Any,
        *,
        event_manifest: Any | None = None,
    ) -> Path:
        import torch

        path = self.path(namespace, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": {
                **self.metadata,
                "event_manifest_hash": (
                    stable_hash({"manifest": event_manifest})
                    if event_manifest is not None else None
                ),
            },
            "value": value,
        }
        temporary = path.with_suffix(path.suffix + ".partial")
        torch.save(payload, temporary)
        temporary.replace(path)
        return path

    def load_torch(
        self,
        namespace: str,
        name: str,
        *,
        event_manifest: Any | None = None,
    ) -> Any | None:
        import torch

        path = self.path(namespace, name)
        if not path.exists():
            return None
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except (OSError, RuntimeError, EOFError):
            return None
        metadata = payload.get("metadata", {})
        if any(metadata.get(key) != value for key, value in self.metadata.items()):
            return None
        expected_manifest = (
            stable_hash({"manifest": event_manifest}) if event_manifest is not None else None
        )
        if metadata.get("event_manifest_hash") != expected_manifest:
            return None
        return payload.get("value")
