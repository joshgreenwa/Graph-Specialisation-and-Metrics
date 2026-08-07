"""Manifest-driven Colab controller for the 30-checkpoint ZINC/QM9 run.

The public notebook in this directory is the intended entry point.  This module keeps the
scientific configuration, fixed checkpoint corpus contract, setup locking, worker selection,
and postflight checks testable without executing a notebook at import time.

One setup run verifies and extracts the archive, warms the two shared PyG datasets, and writes a
commit-pinned notebook per worker.  Production workers are isolated by ``task/seed`` and request
non-retained component results from the canonical runner, which releases score/component state
before carriage while keeping one complete registered ``MethodologyConfig``.
"""

from __future__ import annotations

import contextlib
import dataclasses
import datetime as dt
import gc
import hashlib
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import uuid
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from graph_specialisation_metrics.methodology.protocol import (
    BootstrapPolicy,
    ExecutionPolicy,
    FamilyPolicy,
    MethodologyConfig,
    RunSizes,
)

ARCHIVE_NAME = "zinc_qm9_best_checkpoints.tar"
ARCHIVE_SIDECAR_NAME = f"{ARCHIVE_NAME}.sha256"
ARCHIVE_SHA256 = "1d41d10b5e5c32fa1ce70535de646c762ba3ad4ad3393406c12b64c21cfced22"
ARCHIVE_BYTES = 177_950_720
ARCHIVE_ROOT = "zinc_qm9_best_available_20260807_151904"
EXACT_GLOBAL_BEST = 16
BEST_AVAILABLE = 14
DEFAULT_DRIVE_FOLDER = Path(
    "/content/drive/MyDrive/graph_specialisation_metrics/multi_seed_models"
)

TASKS = (
    "zinc",
    "zinc_1hop",
    "zinc_1hop_vnode",
    "zinc_2hop",
    "zinc_2hop_vnode",
    "qm9_gap_dense",
    "qm9_gap_1hop",
    "qm9_gap_1hop_vnode",
    "qm9_gap_2hop",
    "qm9_gap_2hop_vnode",
)
TRAIN_SEEDS = (0, 1, 2)
PHASES = ("scores", "carriage")

# This is deliberately explicit.  Archive order is never used to decide which architecture a
# checkpoint belongs to.
TASK_RUN_PREFIX = {
    "zinc": "zinc.dense",
    "zinc_1hop": "zinc.1hop",
    "zinc_1hop_vnode": "zinc.1hop_vnode",
    "zinc_2hop": "zinc.2hop",
    "zinc_2hop_vnode": "zinc.2hop_vnode",
    "qm9_gap_dense": "qm9_gap.dense",
    "qm9_gap_1hop": "qm9_gap.1hop",
    "qm9_gap_1hop_vnode": "qm9_gap.1hop_vnode",
    "qm9_gap_2hop": "qm9_gap.2hop",
    "qm9_gap_2hop_vnode": "qm9_gap.2hop_vnode",
}


@dataclass(frozen=True)
class WorkerSpec:
    index: int
    task: str
    seed: int
    run_id: str

    @property
    def key(self) -> str:
        return f"{self.task}:{self.seed}"


WORKERS = tuple(
    WorkerSpec(index, task, seed, f"{TASK_RUN_PREFIX[task]}.s{seed}")
    for index, (task, seed) in enumerate((task, seed) for task in TASKS for seed in TRAIN_SEEDS)
)
WORKER_BY_KEY = {(worker.task, worker.seed): worker for worker in WORKERS}
WORKER_BY_RUN_ID = {worker.run_id: worker for worker in WORKERS}


@dataclass(frozen=True)
class PreparedCorpus:
    drive_folder: Path
    corpus_root: Path
    records: Mapping[str, Mapping[str, Any]]

    def checkpoint_path(self, run_id: str) -> Path:
        if run_id not in self.records:
            raise KeyError(f"unknown checkpoint run ID {run_id!r}")
        return self.corpus_root / "checkpoints" / run_id / "best_available.ckpt"

    @property
    def checkpoints(self) -> dict[str, str]:
        return {worker.key: str(self.checkpoint_path(worker.run_id)) for worker in WORKERS}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=str(path.parent)
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, default=str)
            stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=str(path.parent)
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(value)
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def sha256_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_commit() -> str:
    root = Path(__file__).resolve().parents[2]
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def resolve_worker(
    worker_index: int | None = None,
    *,
    task: str | None = None,
    seed: int | None = None,
) -> WorkerSpec:
    """Resolve either a 0-based worker index or an explicit task/seed pair."""

    explicit = task is not None or seed is not None
    if explicit:
        if task is None or seed is None:
            raise ValueError("explicit worker selection requires both task and seed")
        try:
            return WORKER_BY_KEY[(str(task), int(seed))]
        except KeyError as error:
            raise ValueError(
                f"unknown worker {task!r}:seed{seed}; expected one of "
                f"{[worker.key for worker in WORKERS]}"
            ) from error
    if worker_index is None:
        raise ValueError("set WORKER_INDEX or provide both task and seed")
    index = int(worker_index)
    if not 0 <= index < len(WORKERS):
        raise ValueError(f"WORKER_INDEX must be in [0, {len(WORKERS) - 1}], got {index}")
    return WORKERS[index]


def _read_sidecar(sidecar: Path) -> tuple[str, str]:
    parts = sidecar.read_text(encoding="utf-8").strip().split()
    if len(parts) != 2:
        raise ValueError(f"malformed SHA-256 sidecar {sidecar}: expected HASH FILENAME")
    return parts[0].lower(), parts[1].lstrip("*")


def verify_archive_identity(drive_folder: Path) -> Path:
    archive = Path(drive_folder) / ARCHIVE_NAME
    sidecar = Path(drive_folder) / ARCHIVE_SIDECAR_NAME
    if not archive.is_file() or not sidecar.is_file():
        raise FileNotFoundError(
            f"Upload {ARCHIVE_NAME} and {ARCHIVE_SIDECAR_NAME} directly into "
            f"{Path(drive_folder)} before setup"
        )
    digest, filename = _read_sidecar(sidecar)
    if filename != ARCHIVE_NAME or digest != ARCHIVE_SHA256:
        raise RuntimeError(
            f"sidecar does not identify the registered corpus: got {digest} {filename}, "
            f"expected {ARCHIVE_SHA256} {ARCHIVE_NAME}"
        )
    size = archive.stat().st_size
    if size != ARCHIVE_BYTES:
        raise RuntimeError(f"archive size mismatch: got {size} bytes, expected {ARCHIVE_BYTES}")
    actual = sha256_file(archive)
    if actual != ARCHIVE_SHA256:
        raise RuntimeError(f"archive SHA-256 mismatch: got {actual}, expected {ARCHIVE_SHA256}")
    return archive


def _compact_manifest_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "run_id": str(record["run_id"]),
        "archive_checkpoint": str(record["archive_checkpoint"]),
        "source_checkpoint_sha256": str(record["source_checkpoint_sha256"]).lower(),
        "source_checkpoint_bytes": int(record["source_checkpoint_bytes"]),
        "exact_global_best": bool(record["exact_global_best"]),
        "selected_epoch": int(record["selected_epoch"]),
        "selected_val_mae": float(record["selected_val_mae"]),
        "selected_test_mae": float(record["selected_test_mae"]),
    }


def validate_archive_manifest(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    if len(records) != len(WORKERS):
        raise RuntimeError(f"archive manifest has {len(records)} runs; expected {len(WORKERS)}")
    compact: dict[str, dict[str, Any]] = {}
    for raw in records:
        record = _compact_manifest_record(raw)
        run_id = record["run_id"]
        if run_id in compact:
            raise RuntimeError(f"duplicate run_id in archive manifest: {run_id}")
        if run_id not in WORKER_BY_RUN_ID:
            raise RuntimeError(f"unexpected run_id in archive manifest: {run_id}")
        expected_member = f"checkpoints/{run_id}/best_available.ckpt"
        if record["archive_checkpoint"] != expected_member:
            raise RuntimeError(
                f"checkpoint member mismatch for {run_id}: "
                f"{record['archive_checkpoint']!r} != {expected_member!r}"
            )
        digest = record["source_checkpoint_sha256"]
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise RuntimeError(f"invalid checkpoint SHA-256 for {run_id}: {digest!r}")
        if record["source_checkpoint_bytes"] <= 0:
            raise RuntimeError(f"invalid checkpoint byte count for {run_id}")
        compact[run_id] = record
    missing = sorted(set(WORKER_BY_RUN_ID) - set(compact))
    if missing:
        raise RuntimeError(f"archive manifest is missing runs: {missing}")
    exact = sum(bool(record["exact_global_best"]) for record in compact.values())
    if exact != EXACT_GLOBAL_BEST or len(compact) - exact != BEST_AVAILABLE:
        raise RuntimeError(
            f"checkpoint-selection identity mismatch: exact={exact}, available={len(compact) - exact}; "
            f"expected {EXACT_GLOBAL_BEST}/{BEST_AVAILABLE}"
        )
    return compact


def read_archive_manifest(archive: Path) -> dict[str, dict[str, Any]]:
    expected = f"{ARCHIVE_ROOT}/manifest.json"
    with tarfile.open(archive, "r:") as bundle:
        names = [member.name for member in bundle.getmembers()]
        if len(names) != len(set(names)):
            raise RuntimeError("archive contains duplicate member names")
        try:
            member = bundle.getmember(expected)
        except KeyError as error:
            raise RuntimeError(f"archive is missing {expected}") from error
        stream = bundle.extractfile(member)
        if stream is None:
            raise RuntimeError(f"archive manifest {expected} is not a regular file")
        records = json.load(stream)
    if not isinstance(records, list):
        raise TypeError("archive manifest must be a JSON list")
    return validate_archive_manifest(records)


def _safe_member_path(destination: Path, member_name: str) -> Path:
    relative = PurePosixPath(member_name)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise RuntimeError(f"unsafe archive member path: {member_name!r}")
    if relative.parts[0] != ARCHIVE_ROOT:
        raise RuntimeError(f"archive member escapes registered root: {member_name!r}")
    return destination.joinpath(*relative.parts)


def safe_extract_archive(archive: Path, destination: Path) -> None:
    """Extract only ordinary directories/files, using atomic replacement per file."""

    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:") as bundle:
        for member in bundle.getmembers():
            target = _safe_member_path(destination, member.name)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise RuntimeError(
                    f"archive member {member.name!r} is not a regular file/directory"
                )
            source = bundle.extractfile(member)
            if source is None:
                raise RuntimeError(f"could not read archive member {member.name!r}")
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.partial")
            try:
                with temporary.open("xb") as output:
                    shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
                os.replace(temporary, target)
            except BaseException:
                with contextlib.suppress(FileNotFoundError):
                    temporary.unlink()
                raise


def _corpus_marker_path(drive_folder: Path) -> Path:
    return Path(drive_folder) / "corpus" / "corpus_ready.json"


def _validate_extracted_checkpoints(
    corpus_root: Path, records: Mapping[str, Mapping[str, Any]], *, hash_all: bool
) -> None:
    for worker in WORKERS:
        record = records[worker.run_id]
        checkpoint = corpus_root / "checkpoints" / worker.run_id / "best_available.ckpt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"extracted checkpoint is missing: {checkpoint}")
        size = checkpoint.stat().st_size
        if size != int(record["source_checkpoint_bytes"]):
            raise RuntimeError(
                f"checkpoint byte mismatch for {worker.run_id}: got {size}, "
                f"expected {record['source_checkpoint_bytes']}"
            )
        if hash_all:
            actual = sha256_file(checkpoint)
            expected = str(record["source_checkpoint_sha256"])
            if actual != expected:
                raise RuntimeError(
                    f"checkpoint SHA-256 mismatch for {worker.run_id}: {actual} != {expected}"
                )


def prepare_corpus(drive_folder: Path) -> PreparedCorpus:
    """Verify the fixed tarball, extract it once on Drive, and hash all 30 checkpoints."""

    drive_folder = Path(drive_folder)
    archive = verify_archive_identity(drive_folder)
    records = read_archive_manifest(archive)
    extraction_root = drive_folder / "corpus" / ARCHIVE_SHA256
    corpus_root = extraction_root / ARCHIVE_ROOT
    safe_extract_archive(archive, extraction_root)
    _validate_extracted_checkpoints(corpus_root, records, hash_all=True)
    marker = {
        "schema": "zinc-qm9-checkpoint-corpus-v1",
        "created_at": utc_now(),
        "archive": {
            "name": ARCHIVE_NAME,
            "bytes": ARCHIVE_BYTES,
            "sha256": ARCHIVE_SHA256,
            "root": ARCHIVE_ROOT,
            "exact_global_best": EXACT_GLOBAL_BEST,
            "best_available": BEST_AVAILABLE,
        },
        "corpus_root": str(corpus_root),
        "records": [records[worker.run_id] for worker in WORKERS],
    }
    _atomic_json(_corpus_marker_path(drive_folder), marker)
    return PreparedCorpus(drive_folder, corpus_root, records)


def load_prepared_corpus(
    drive_folder: Path, *, selected: WorkerSpec | None = None
) -> PreparedCorpus:
    marker_path = _corpus_marker_path(Path(drive_folder))
    if not marker_path.is_file():
        raise RuntimeError(
            f"checkpoint corpus is not prepared ({marker_path} is missing); run MODE='setup' once"
        )
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    identity = marker.get("archive", {})
    expected_identity = {
        "name": ARCHIVE_NAME,
        "bytes": ARCHIVE_BYTES,
        "sha256": ARCHIVE_SHA256,
        "root": ARCHIVE_ROOT,
        "exact_global_best": EXACT_GLOBAL_BEST,
        "best_available": BEST_AVAILABLE,
    }
    if identity != expected_identity:
        raise RuntimeError(f"prepared corpus identity mismatch: {identity!r}")
    records = validate_archive_manifest(marker.get("records", []))
    corpus_root = Path(marker.get("corpus_root", ""))
    corpus = PreparedCorpus(Path(drive_folder), corpus_root, records)
    _validate_extracted_checkpoints(corpus_root, records, hash_all=False)
    if selected is not None:
        checkpoint = corpus.checkpoint_path(selected.run_id)
        actual = sha256_file(checkpoint)
        expected = str(records[selected.run_id]["source_checkpoint_sha256"])
        if actual != expected:
            raise RuntimeError(
                f"selected checkpoint SHA-256 mismatch for {selected.run_id}: {actual} != {expected}"
            )
    return corpus


def _lock_path(drive_folder: Path, name: str) -> Path:
    return Path(drive_folder) / "_locks" / f"{name}.lock"


@contextlib.contextmanager
def drive_lock(
    drive_folder: Path,
    name: str,
    *,
    reclaim: bool = False,
) -> Iterator[Path]:
    """Claim an exact Drive directory; a crashed claim requires explicit reclamation."""

    lock = _lock_path(Path(drive_folder), name)
    lock.parent.mkdir(parents=True, exist_ok=True)
    if reclaim and lock.exists():
        claim = lock / "claim.json"
        extras = [path.name for path in lock.iterdir() if path.name != "claim.json"]
        if extras:
            raise RuntimeError(f"refusing to reclaim non-empty lock {lock}: extras={extras}")
        with contextlib.suppress(FileNotFoundError):
            claim.unlink()
        lock.rmdir()
    try:
        lock.mkdir()
    except FileExistsError as error:
        claim = lock / "claim.json"
        detail = claim.read_text(encoding="utf-8") if claim.is_file() else "(no claim record)"
        raise RuntimeError(
            f"Drive claim already exists for {name!r}: {lock}\n{detail}\n"
            "Do not run the same worker twice. If its runtime is gone, set "
            "RECLAIM_STALE_LOCK=True once."
        ) from error
    claim = lock / "claim.json"
    claim_record = {
        "name": name,
        "claimed_at": utc_now(),
        "host": platform.node(),
        "pid": os.getpid(),
        "token": uuid.uuid4().hex,
    }
    _atomic_json(claim, claim_record)
    try:
        yield lock
    finally:
        # Only remove the exact claim this context created.
        current = None
        with contextlib.suppress(FileNotFoundError, json.JSONDecodeError):
            current = json.loads(claim.read_text(encoding="utf-8"))
        if current == claim_record:
            with contextlib.suppress(FileNotFoundError):
                claim.unlink()
            with contextlib.suppress(FileNotFoundError, OSError):
                lock.rmdir()


def _inventory(root: Path) -> list[dict[str, Any]]:
    return [
        {"path": str(path.relative_to(root)), "bytes": path.stat().st_size}
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]


def warm_shared_datasets(drive_folder: Path) -> Mapping[str, Any]:
    """Download/process each underlying PyG dataset exactly once in the shared folder."""

    from torch_geometric.datasets import QM9, ZINC

    datasets_root = Path(drive_folder) / "datasets"
    zinc_root = datasets_root / "zinc"
    qm9_root = datasets_root / "qm9"
    zinc_sizes = {
        split: len(ZINC(str(zinc_root), subset=True, split=split))
        for split in ("train", "val", "test")
    }
    qm9_size = len(QM9(str(qm9_root)))
    inventories = {
        "zinc": _inventory(zinc_root),
        "qm9": _inventory(qm9_root),
    }
    if not inventories["zinc"] or not inventories["qm9"]:
        raise RuntimeError("dataset warmup completed without persistent dataset files")
    marker = {
        "schema": "zinc-qm9-shared-datasets-v1",
        "created_at": utc_now(),
        "roots": {"zinc": str(zinc_root), "qm9": str(qm9_root)},
        "sizes": {"zinc": zinc_sizes, "qm9": qm9_size},
        "files": inventories,
    }
    _atomic_json(Path(drive_folder) / "datasets_ready.json", marker)
    return marker


def validate_shared_datasets(drive_folder: Path) -> Mapping[str, Any]:
    marker_path = Path(drive_folder) / "datasets_ready.json"
    if not marker_path.is_file():
        raise RuntimeError(
            f"shared datasets are not prepared ({marker_path} is missing); run MODE='setup' once"
        )
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("schema") != "zinc-qm9-shared-datasets-v1":
        raise RuntimeError(f"unexpected dataset marker schema in {marker_path}")
    for dataset in ("zinc", "qm9"):
        root = Path(marker["roots"][dataset])
        records = marker["files"][dataset]
        if not records:
            raise RuntimeError(f"empty file inventory for {dataset}")
        for record in records:
            path = root / record["path"]
            if not path.is_file() or path.stat().st_size != int(record["bytes"]):
                raise RuntimeError(
                    f"prepared {dataset} file is missing/truncated: {path}; rerun setup"
                )
    return marker


def generate_worker_notebooks(
    template: Path,
    destination: Path,
    *,
    revision: str,
) -> tuple[Path, ...]:
    """Create 30 pre-indexed, production-mode notebook copies on Drive."""

    payload = json.loads(Path(template).read_text(encoding="utf-8"))
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    generated = []
    for worker in WORKERS:
        copy = json.loads(json.dumps(payload))
        replacements = {
            "MODE = ": 'MODE = "worker"  # @param ["setup", "preflight", "smoke", "worker", "status", "finalize"]\n',
            "WORKER_INDEX = ": f'WORKER_INDEX = {worker.index}  # @param {{type:"integer"}}\n',
            "REPO_REVISION = ": f'REPO_REVISION = "{revision}"\n',
        }
        counts = {prefix: 0 for prefix in replacements}
        for cell in copy.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            lines = cell.get("source", [])
            for position, line in enumerate(lines):
                for prefix, replacement in replacements.items():
                    if line.startswith(prefix):
                        lines[position] = replacement
                        counts[prefix] += 1
        if any(count != 1 for count in counts.values()):
            raise RuntimeError(f"notebook template controls were not unique: {counts}")
        filename = f"worker_{worker.index:02d}_{worker.task}_seed{worker.seed}.ipynb"
        target = destination / filename
        _atomic_text(target, json.dumps(copy, indent=1) + "\n")
        generated.append(target)
    return tuple(generated)


def _setup_marker_path(drive_folder: Path) -> Path:
    return Path(drive_folder) / "setup_ready.json"


def setup_drive(
    drive_folder: Path,
    *,
    notebook_template: Path,
    reclaim_stale_lock: bool = False,
) -> Mapping[str, Any]:
    drive_folder = Path(drive_folder)
    drive_folder.mkdir(parents=True, exist_ok=True)
    with drive_lock(drive_folder, "dataset_and_corpus_setup", reclaim=reclaim_stale_lock):
        corpus = prepare_corpus(drive_folder)
        datasets = warm_shared_datasets(drive_folder)
        commit = repository_commit()
        notebooks = generate_worker_notebooks(
            notebook_template,
            drive_folder / "worker_notebooks",
            revision=commit,
        )
        marker = {
            "schema": "zinc-qm9-canonical-colab-setup-v1",
            "created_at": utc_now(),
            "repository_commit": commit,
            "archive_sha256": ARCHIVE_SHA256,
            "corpus_root": str(corpus.corpus_root),
            "datasets": datasets["roots"],
            "worker_notebooks": [str(path) for path in notebooks],
        }
        _atomic_json(_setup_marker_path(drive_folder), marker)
    return marker


def require_ready_setup(drive_folder: Path) -> Mapping[str, Any]:
    marker_path = _setup_marker_path(Path(drive_folder))
    if not marker_path.is_file():
        raise RuntimeError(
            f"setup gate is missing ({marker_path}); run MODE='setup' before parallel workers"
        )
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("schema") != "zinc-qm9-canonical-colab-setup-v1":
        raise RuntimeError(f"unexpected setup schema in {marker_path}")
    if marker.get("archive_sha256") != ARCHIVE_SHA256:
        raise RuntimeError("setup marker belongs to a different checkpoint archive")
    current = repository_commit()
    if marker.get("repository_commit") != current:
        raise RuntimeError(
            f"repository commit differs from setup: current={current}, "
            f"setup={marker.get('repository_commit')}. Open the generated commit-pinned "
            "worker notebooks or rerun setup."
        )
    validate_shared_datasets(Path(drive_folder))
    return marker


def auto_graphs_per_batch(
    *, device_name: str | None = None, total_memory_bytes: int | None = None
) -> int:
    """Return a high-throughput starting batch; the runner halves it on CUDA OOM."""

    if device_name is None or total_memory_bytes is None:
        try:
            import torch

            if not torch.cuda.is_available():
                return 4
            properties = torch.cuda.get_device_properties(0)
            device_name = str(properties.name)
            total_memory_bytes = int(properties.total_memory)
        except (ImportError, RuntimeError):
            return 4
    gib = int(total_memory_bytes) / float(1024**3)
    name = str(device_name).lower()
    if ("a100" in name or "h100" in name) and gib >= 75:
        return 48
    if gib >= 70:
        return 48
    if gib >= 38:
        return 24
    if gib >= 20:
        return 12
    if gib >= 14:
        return 8
    return 4


def build_production_config(
    drive_folder: Path,
    corpus: PreparedCorpus,
    *,
    graphs_per_batch: int | None = None,
    accelerator: str = "cuda:0",
    strict_audits: bool = False,
) -> MethodologyConfig:
    """Build the complete 10-task/3-seed contract used by every production worker."""

    drive_folder = Path(drive_folder)
    datasets = {
        "zinc": str(drive_folder / "datasets" / "zinc"),
        "qm9": str(drive_folder / "datasets" / "qm9"),
    }
    task_overrides = {
        task: {
            "drive_dir": str(drive_folder / "runtime" / task),
            "dataset_dir": datasets["zinc" if task.startswith("zinc") else "qm9"],
        }
        for task in TASKS
    }
    batch = int(graphs_per_batch or auto_graphs_per_batch())
    config = MethodologyConfig(
        output_dir=str(drive_folder / "canonical_outputs"),
        tasks=TASKS,
        train_seeds=TRAIN_SEEDS,
        task_train_seeds={},
        phases=PHASES,
        sizes=RunSizes(),
        bootstrap=BootstrapPolicy(
            rng_seed=17_071,
            replicates=2_000,
            resample_source=True,
        ),
        families=FamilyPolicy(),
        execution=ExecutionPolicy(
            graphs_per_batch=batch,
            oom_backoff=True,
            replica_pair_budget=None,
            jacobian_output_chunk=8,
            progress_heartbeat_seconds=30.0,
        ),
        analysis_seed=31_415,
        accelerator=accelerator,
        num_threads=4,
        checkpoints=corpus.checkpoints,
        task_overrides=task_overrides,
        skip_install=True,
        resume=True,
        force=False,
        strict_audits=bool(strict_audits),
        compute_beneficial_carriage=True,
    )
    config.validate()
    if len(config.checkpoints) != len(WORKERS):
        raise RuntimeError("production configuration must contain all 30 explicit checkpoints")
    return config


def build_smoke_config(config: MethodologyConfig) -> MethodologyConfig:
    smoke = dataclasses.replace(
        config,
        output_dir=str(Path(config.output_dir).parent / "smoke_outputs"),
        sizes=RunSizes.smoke(),
        execution=dataclasses.replace(
            config.execution, graphs_per_batch=min(3, config.execution.graphs_per_batch)
        ),
    )
    smoke.validate()
    return smoke


def release_component_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            with contextlib.suppress(Exception):
                torch.cuda.synchronize()
            with contextlib.suppress(Exception):
                torch.cuda.empty_cache()
            ipc_collect = getattr(torch.cuda, "ipc_collect", None)
            if callable(ipc_collect):
                with contextlib.suppress(Exception):
                    ipc_collect()
    except ImportError:
        pass


def validate_worker_outputs(
    config: MethodologyConfig,
    worker: WorkerSpec,
    *,
    expected_graphs: int,
    expected_checkpoint_sha256: str,
) -> Mapping[str, Any]:
    from graph_specialisation_metrics.methodology.cache import load_cache_artifact_file

    root = Path(config.output_dir) / worker.task / f"seed_{worker.seed}"
    paths = {
        "scores": root / "cache" / "scores" / "raw.pt",
        "carriage": root / "cache" / "carriage" / "fields.pt",
    }
    contract_records: dict[str, dict[str, Any]] = {}
    for stage, path in paths.items():
        artifact = load_cache_artifact_file(path)
        try:
            contract = artifact.metadata["contract"]
            if contract["task"] != worker.task or int(contract["train_seed"]) != worker.seed:
                raise RuntimeError(f"{stage} cache belongs to another worker: {contract}")
            if contract["protocol_fingerprint"] != config.fingerprint:
                raise RuntimeError(f"{stage} protocol fingerprint mismatch")
            if contract["checkpoint_sha256"] != expected_checkpoint_sha256:
                raise RuntimeError(f"{stage} checkpoint SHA-256 mismatch")
            for channel in ("semantic", "structural"):
                field = "graph_scores" if stage == "scores" else "graph_fields"
                count = len(artifact.value["channels"][channel][field])
                if count != int(expected_graphs):
                    raise RuntimeError(
                        f"{worker.key} {stage}/{channel} has {count} graphs; "
                        f"expected {expected_graphs}"
                    )
            contract_records[stage] = {
                "protocol_fingerprint": str(contract["protocol_fingerprint"]),
                "checkpoint_sha256": str(contract["checkpoint_sha256"]),
            }
        finally:
            del artifact
            gc.collect()
    if (
        contract_records["scores"]["protocol_fingerprint"]
        != contract_records["carriage"]["protocol_fingerprint"]
    ):
        raise RuntimeError("score/carriage protocol fingerprints differ")
    partials = [str(path) for path in root.glob("cache/**/*.partial")]
    if partials:
        raise RuntimeError(f"unfinished cache writes remain: {partials}")
    return {
        "worker": dataclasses.asdict(worker),
        "protocol_fingerprint": config.fingerprint,
        "checkpoint_sha256": expected_checkpoint_sha256,
        "graphs": int(expected_graphs),
        "completed_at": utc_now(),
    }


def _restore_complete_worker_protocol(config: MethodologyConfig, worker: WorkerSpec) -> None:
    record = config.record()
    record.update(
        {
            "repository_commit": repository_commit(),
            "execution_mode": "isolated-seed-worker",
            "worker_task": worker.task,
            "worker_seed": worker.seed,
            "component_result_retention": False,
        }
    )
    root = Path(config.output_dir) / worker.task / f"seed_{worker.seed}"
    _atomic_json(root / "protocol.json", record)


def run_selected_worker(
    config: MethodologyConfig,
    corpus: PreparedCorpus,
    worker: WorkerSpec,
    *,
    reclaim_stale_lock: bool = False,
    expected_graphs: int = 48,
    lock_prefix: str = "production",
) -> Mapping[str, Any]:
    """Run the complete worker while dropping component results as soon as they are cached."""

    from graph_specialisation_metrics.methodology.runner import run_worker

    expected_digest = str(corpus.records[worker.run_id]["source_checkpoint_sha256"])
    claim = f"{lock_prefix}_{worker.task}_seed{worker.seed}"
    with drive_lock(corpus.drive_folder, claim, reclaim=reclaim_stale_lock):
        result = None
        try:
            result = run_worker(
                config,
                worker.task,
                worker.seed,
                retain_results=False,
            )
        finally:
            if result is not None:
                del result
            release_component_memory()
        _restore_complete_worker_protocol(config, worker)
        completion = validate_worker_outputs(
            config,
            worker,
            expected_graphs=expected_graphs,
            expected_checkpoint_sha256=expected_digest,
        )
        root = Path(config.output_dir) / worker.task / f"seed_{worker.seed}"
        _atomic_json(root / "worker_complete.json", completion)
    return completion


def completion_status(drive_folder: Path) -> list[dict[str, Any]]:
    output = Path(drive_folder) / "canonical_outputs"
    rows = []
    for worker in WORKERS:
        root = output / worker.task / f"seed_{worker.seed}"
        rows.append(
            {
                "index": worker.index,
                "task": worker.task,
                "seed": worker.seed,
                "scores": (root / "cache" / "scores" / "raw.pt").is_file(),
                "carriage": (root / "cache" / "carriage" / "fields.pt").is_file(),
                "validated": (root / "worker_complete.json").is_file(),
            }
        )
    return rows


def _print_status(rows: Sequence[Mapping[str, Any]]) -> None:
    print("index  task                   seed  scores  carriage  validated", flush=True)
    for row in rows:
        print(
            f"{int(row['index']):>5}  {row['task']!s:<21}  {int(row['seed']):>4}  "
            f"{bool(row['scores'])!s:<6}  {bool(row['carriage'])!s:<8}  "
            f"{bool(row['validated'])}",
            flush=True,
        )
    complete = sum(bool(row["validated"]) for row in rows)
    print(f"[status] {complete}/{len(rows)} workers validated", flush=True)


def dependency_stack_ready() -> bool:
    try:
        import ogb  # noqa: F401
        import pytorch_lightning  # noqa: F401
        import torch_geometric
        import yacs  # noqa: F401

        return str(torch_geometric.__version__) == "2.2.0"
    except Exception:  # noqa: BLE001 - a partially incompatible binary stack is "not ready"
        return False


def ensure_runtime_dependencies() -> None:
    if dependency_stack_ready():
        print("[deps] compatible GRIT/PyG stack already present", flush=True)
    else:
        from graph_specialisation_metrics.carriage import env

        env.install_dependencies(pyg_version="2.2.0")
        # The readiness probe can import a preinstalled but incompatible package before pip
        # replaces it. Never let those old module objects survive into dataset/model setup.
        runtime_prefixes = (
            "ogb",
            "pytorch_lightning",
            "torch_cluster",
            "torch_geometric",
            "torch_scatter",
            "torch_sparse",
            "torchmetrics",
            "yacs",
        )
        for name in tuple(sys.modules):
            if any(name == prefix or name.startswith(f"{prefix}.") for prefix in runtime_prefixes):
                del sys.modules[name]
        importlib.invalidate_caches()
    from graph_specialisation_metrics.carriage import env

    env.apply_compat_patches()


def require_requested_accelerator(accelerator: str) -> Mapping[str, Any]:
    import torch

    requested = str(accelerator)
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "This worker requested CUDA but Colab has no GPU. Select a GPU runtime before "
            "running production or smoke mode."
        )
    if not requested.startswith("cuda"):
        return {"accelerator": requested}
    index = torch.device(requested).index or 0
    properties = torch.cuda.get_device_properties(index)
    record = {
        "accelerator": requested,
        "device_name": str(properties.name),
        "total_memory_gib": round(float(properties.total_memory) / 1024**3, 2),
    }
    print(f"[gpu] {json.dumps(record, sort_keys=True)}", flush=True)
    return record


def preflight(
    drive_folder: Path,
    worker: WorkerSpec,
    *,
    graphs_per_batch: int | None = None,
) -> Mapping[str, Any]:
    require_ready_setup(drive_folder)
    corpus = load_prepared_corpus(drive_folder, selected=worker)
    config = build_production_config(drive_folder, corpus, graphs_per_batch=graphs_per_batch)
    record = {
        "worker": dataclasses.asdict(worker),
        "checkpoint": config.checkpoints[worker.key],
        "checkpoint_sha256": corpus.records[worker.run_id]["source_checkpoint_sha256"],
        "protocol_fingerprint": config.fingerprint,
        "graphs_per_batch": config.execution.graphs_per_batch,
        "oom_backoff": config.execution.oom_backoff,
        "phases": list(config.phases),
    }
    print(json.dumps(record, indent=2), flush=True)
    return record


def run_frontend(
    *,
    mode: str,
    drive_folder: str | Path = DEFAULT_DRIVE_FOLDER,
    worker_index: int = 0,
    task: str | None = None,
    seed: int | None = None,
    graphs_per_batch: int | None = None,
    accelerator: str = "cuda:0",
    strict_audits: bool = False,
    reclaim_stale_lock: bool = False,
) -> Any:
    """Execute a notebook mode after Drive is mounted and this repository is imported."""

    mode = str(mode).strip().lower()
    drive_folder = Path(drive_folder)
    template = Path(__file__).with_name("zinc_qm9_canonical_worker_colab.ipynb")
    if mode == "setup":
        ensure_runtime_dependencies()
        result = setup_drive(
            drive_folder,
            notebook_template=template,
            reclaim_stale_lock=reclaim_stale_lock,
        )
        print(
            f"[setup:complete] generated {len(result['worker_notebooks'])} workers under "
            f"{drive_folder / 'worker_notebooks'}",
            flush=True,
        )
        _print_status(completion_status(drive_folder))
        return result
    if mode == "status":
        rows = completion_status(drive_folder)
        _print_status(rows)
        return rows

    worker = resolve_worker(worker_index, task=task, seed=seed)
    if mode == "preflight":
        return preflight(drive_folder, worker, graphs_per_batch=graphs_per_batch)

    require_ready_setup(drive_folder)
    corpus = load_prepared_corpus(drive_folder, selected=worker)
    config = build_production_config(
        drive_folder,
        corpus,
        graphs_per_batch=graphs_per_batch,
        accelerator=accelerator,
        strict_audits=strict_audits,
    )
    if mode == "finalize":
        from graph_specialisation_metrics.methodology.runner import (
            finalize_measurement_run,
        )

        result = finalize_measurement_run(config)
        _print_status(completion_status(drive_folder))
        return result
    if mode not in {"smoke", "worker"}:
        raise ValueError("MODE must be one of setup, preflight, smoke, worker, status, finalize")
    ensure_runtime_dependencies()
    require_requested_accelerator(config.accelerator)
    if mode == "smoke":
        smoke = build_smoke_config(config)
        return run_selected_worker(
            smoke,
            corpus,
            worker,
            reclaim_stale_lock=reclaim_stale_lock,
            expected_graphs=smoke.sizes.discovery_graphs,
            lock_prefix="smoke",
        )
    return run_selected_worker(
        config,
        corpus,
        worker,
        reclaim_stale_lock=reclaim_stale_lock,
        expected_graphs=config.sizes.discovery_graphs,
        lock_prefix="production",
    )


__all__ = [
    "ARCHIVE_BYTES",
    "ARCHIVE_NAME",
    "ARCHIVE_ROOT",
    "ARCHIVE_SHA256",
    "ARCHIVE_SIDECAR_NAME",
    "DEFAULT_DRIVE_FOLDER",
    "PHASES",
    "TASKS",
    "TRAIN_SEEDS",
    "WORKERS",
    "PreparedCorpus",
    "WorkerSpec",
    "auto_graphs_per_batch",
    "build_production_config",
    "build_smoke_config",
    "completion_status",
    "generate_worker_notebooks",
    "load_prepared_corpus",
    "preflight",
    "prepare_corpus",
    "resolve_worker",
    "run_frontend",
    "run_selected_worker",
    "setup_drive",
    "validate_archive_manifest",
    "verify_archive_identity",
]
