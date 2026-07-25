"""Small, dependency-free helpers for resumable graph-wise analyses.

Progress files are cumulative snapshots written atomically.  A canonical request fingerprint
prevents a snapshot made for a different checkpoint, graph sample, or estimator configuration
from ever being reused silently.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


FORMAT_VERSION = 1


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default)


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def checkpoint_identity(path) -> dict:
    """Stable-enough identity for rejecting progress from another model checkpoint."""
    p = Path(path).expanduser().resolve()
    st = p.stat()
    return {"path": str(p), "size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}


def save_progress(path, *, fingerprint: dict, next_index: int, rng_state: dict,
                  arrays: dict[str, np.ndarray], scalars: dict) -> None:
    """Atomically replace ``path`` with a compressed, pickle-free progress snapshot."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    payload = {
        "format_version": np.asarray(FORMAT_VERSION, dtype=np.int64),
        "fingerprint": np.asarray(canonical_json(fingerprint)),
        "next_index": np.asarray(int(next_index), dtype=np.int64),
        "rng_state": np.asarray(canonical_json(rng_state)),
        "scalars": np.asarray(canonical_json(scalars)),
        **{name: np.asarray(value) for name, value in arrays.items()},
    }
    with tmp.open("wb") as fh:
        np.savez_compressed(fh, **payload)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def load_progress(path, *, fingerprint: dict) -> dict | None:
    """Return a validated snapshot, or ``None`` for absent/stale/corrupt progress."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            if int(z["format_version"]) != FORMAT_VERSION:
                return None
            if str(z["fingerprint"].item()) != canonical_json(fingerprint):
                return None
            return {
                "next_index": int(z["next_index"]),
                "rng_state": json.loads(str(z["rng_state"].item())),
                "scalars": json.loads(str(z["scalars"].item())),
                "arrays": {
                    name: np.asarray(z[name])
                    for name in z.files
                    if name not in {"format_version", "fingerprint", "next_index",
                                    "rng_state", "scalars"}
                },
            }
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def concat(parts, *, dtype) -> np.ndarray:
    """Concatenate accumulator parts with a typed empty fallback."""
    nonempty = [np.asarray(x) for x in parts if np.asarray(x).size]
    return np.concatenate(nonempty).astype(dtype, copy=False) if nonempty else np.asarray([], dtype=dtype)
