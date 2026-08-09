"""Two-lane Colab controller for the 30 Peptides-func/struct checkpoints.

Each dataset lane runs its fifteen architecture/seed workers sequentially.  Both lanes use the
same complete 30-worker scientific configuration, so their score and carriage artifacts form one
finalizable population.  Checkpoints and resumable caches live in the uploaded Drive folder.
"""

from __future__ import annotations

import contextlib
import dataclasses
import gc
import json
import math
import os
import shutil
import tarfile
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from experiments.methodology.zinc_qm9_canonical_colab_worker import (
    _atomic_json,
    drive_lock,
    ensure_runtime_dependencies,
    release_component_memory,
    repository_commit,
    require_requested_accelerator,
    sha256_file,
    utc_now,
)
from graph_specialisation_metrics.methodology.protocol import (
    BootstrapPolicy,
    ExecutionPolicy,
    FamilyPolicy,
    MethodologyConfig,
    RunSizes,
)

ARCHIVE_NAME = "peptides_func_struct_best_checkpoints.tar"
ARCHIVE_SIDECAR_NAME = f"{ARCHIVE_NAME}.sha256"
ARCHIVE_SHA256 = "d6e5923249326f0d2498d7f5ca996ec2a25c4d213c41cbfdf9c7a28224ad3456"
ARCHIVE_BYTES = 162_631_680
ARCHIVE_ROOT = "peptides_best_available_20260809_153204"
DEFAULT_DRIVE_FOLDER = Path(
    "/content/drive/MyDrive/graph_specialisation_metrics/multi_seed_models/"
    "peptides_func_struct_checkpoints"
)
COMPLETION_SCHEMA = "peptides-func-struct-canonical-worker-complete-v2"
CORPUS_SCHEMA = "peptides-func-struct-checkpoint-corpus-v1"

TASKS = (
    "peptides_func_dense",
    "peptides_func_1hop",
    "peptides_func_1hop_vnode",
    "peptides_func_2hop",
    "peptides_func_2hop_vnode",
    "peptides_struct_dense",
    "peptides_struct_1hop",
    "peptides_struct_1hop_vnode",
    "peptides_struct_2hop",
    "peptides_struct_2hop_vnode",
)
TRAIN_SEEDS = (0, 1, 2)
PHASES = ("scores", "carriage")
TASK_RUN_PREFIX = {
    "peptides_func_dense": "peptides_func.dense",
    "peptides_func_1hop": "peptides_func.1hop",
    "peptides_func_1hop_vnode": "peptides_func.1hop_vnode",
    "peptides_func_2hop": "peptides_func.2hop",
    "peptides_func_2hop_vnode": "peptides_func.2hop_vnode",
    "peptides_struct_dense": "peptides_struct.dense",
    "peptides_struct_1hop": "peptides_struct.1hop",
    "peptides_struct_1hop_vnode": "peptides_struct.1hop_vnode",
    "peptides_struct_2hop": "peptides_struct.2hop",
    "peptides_struct_2hop_vnode": "peptides_struct.2hop_vnode",
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

    @property
    def dataset(self) -> str:
        return "func" if self.task.startswith("peptides_func_") else "struct"


WORKERS = tuple(
    WorkerSpec(index, task, seed, f"{TASK_RUN_PREFIX[task]}.s{seed}")
    for index, (task, seed) in enumerate((task, seed) for task in TASKS for seed in TRAIN_SEEDS)
)
WORKER_BY_RUN_ID = {worker.run_id: worker for worker in WORKERS}
DATASET_WORKERS = {
    dataset: tuple(worker for worker in WORKERS if worker.dataset == dataset)
    for dataset in ("func", "struct")
}
SEED_LANE_WORKERS = {
    (dataset, seed): tuple(
        worker for worker in WORKERS if worker.dataset == dataset and worker.seed == seed
    )
    for dataset in ("func", "struct")
    for seed in TRAIN_SEEDS
}


@dataclass(frozen=True)
class PreparedCorpus:
    drive_folder: Path
    corpus_root: Path
    records: Mapping[str, Mapping[str, Any]]

    def checkpoint_path(self, run_id: str) -> Path:
        return self.corpus_root / "checkpoints" / run_id / "best_available.ckpt"

    @property
    def checkpoints(self) -> dict[str, str]:
        return {worker.key: str(self.checkpoint_path(worker.run_id)) for worker in WORKERS}


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
            f"Place {ARCHIVE_NAME} and {ARCHIVE_SIDECAR_NAME} directly in {drive_folder}"
        )
    digest, filename = _read_sidecar(sidecar)
    if (digest, filename) != (ARCHIVE_SHA256, ARCHIVE_NAME):
        raise RuntimeError(
            f"sidecar identifies {digest} {filename}; expected {ARCHIVE_SHA256} {ARCHIVE_NAME}"
        )
    if archive.stat().st_size != ARCHIVE_BYTES:
        raise RuntimeError(f"archive has {archive.stat().st_size} bytes; expected {ARCHIVE_BYTES}")
    actual = sha256_file(archive)
    if actual != ARCHIVE_SHA256:
        raise RuntimeError(f"archive SHA-256 mismatch: {actual} != {ARCHIVE_SHA256}")
    return archive


def _compact_manifest_record(raw: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "run_id": str(raw["run_id"]),
        "task": str(raw["task"]),
        "variant": str(raw["variant"]),
        "seed": int(raw["seed"]),
        "archive_checkpoint": str(raw["archive_checkpoint"]),
        "source_checkpoint_sha256": str(raw["source_checkpoint_sha256"]).lower(),
        "source_checkpoint_bytes": int(raw["source_checkpoint_bytes"]),
        "exact_global_best": bool(raw["exact_global_best"]),
        "selected_epoch": int(raw["selected_epoch"]),
        "selected_validation": float(raw["selected_validation"]),
        "selected_test": float(raw["selected_test"]),
        "selection_metric": str(raw["selection_metric"]),
        "selection_direction": str(raw["selection_direction"]),
    }


def validate_archive_manifest(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    if len(records) != len(WORKERS):
        raise RuntimeError(f"archive manifest has {len(records)} rows; expected 30")
    compact: dict[str, dict[str, Any]] = {}
    for raw in records:
        record = _compact_manifest_record(raw)
        run_id = record["run_id"]
        if run_id in compact:
            raise RuntimeError(f"duplicate archive run ID {run_id}")
        worker = WORKER_BY_RUN_ID.get(run_id)
        if worker is None:
            raise RuntimeError(f"unexpected archive run ID {run_id}")
        expected_member = f"checkpoints/{run_id}/best_available.ckpt"
        if record["archive_checkpoint"] != expected_member:
            raise RuntimeError(f"wrong checkpoint member for {run_id}")
        if (record["task"], record["seed"]) != (
            "peptides_func" if worker.dataset == "func" else "peptides_struct",
            worker.seed,
        ):
            raise RuntimeError(f"task/seed mismatch in manifest for {run_id}")
        digest = record["source_checkpoint_sha256"]
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise RuntimeError(f"invalid checkpoint SHA-256 for {run_id}")
        if record["source_checkpoint_bytes"] <= 0 or not record["exact_global_best"]:
            raise RuntimeError(f"invalid checkpoint selection record for {run_id}")
        compact[run_id] = record
    if set(compact) != set(WORKER_BY_RUN_ID):
        raise RuntimeError("archive manifest does not cover the exact 30-worker grid")
    return compact


def read_archive_manifest(archive: Path) -> dict[str, dict[str, Any]]:
    member_name = f"{ARCHIVE_ROOT}/manifest.json"
    with tarfile.open(archive, "r:") as bundle:
        members = bundle.getmembers()
        if len({member.name for member in members}) != len(members):
            raise RuntimeError("archive contains duplicate member names")
        try:
            member = bundle.getmember(member_name)
        except KeyError as error:
            raise RuntimeError(f"archive is missing {member_name}") from error
        stream = bundle.extractfile(member)
        if stream is None:
            raise RuntimeError(f"archive manifest {member_name} is unreadable")
        records = json.load(stream)
    if not isinstance(records, list):
        raise TypeError("archive manifest must be a JSON list")
    return validate_archive_manifest(records)


def _safe_member_path(destination: Path, member_name: str) -> Path:
    relative = PurePosixPath(member_name)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise RuntimeError(f"unsafe archive member path {member_name!r}")
    if relative.parts[0] != ARCHIVE_ROOT:
        raise RuntimeError(f"archive member escapes the registered root: {member_name!r}")
    return destination.joinpath(*relative.parts)


def safe_extract_archive(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:") as bundle:
        for member in bundle.getmembers():
            target = _safe_member_path(destination, member.name)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise RuntimeError(f"archive member {member.name!r} is not a regular file")
            source = bundle.extractfile(member)
            if source is None:
                raise RuntimeError(f"cannot read archive member {member.name!r}")
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
    return Path(drive_folder) / "corpus" / "peptides_corpus_ready.json"


def _validate_extracted_checkpoints(
    corpus_root: Path,
    records: Mapping[str, Mapping[str, Any]],
    *,
    hash_all: bool,
) -> None:
    for worker in WORKERS:
        checkpoint = corpus_root / "checkpoints" / worker.run_id / "best_available.ckpt"
        record = records[worker.run_id]
        if not checkpoint.is_file():
            raise FileNotFoundError(f"missing extracted checkpoint {checkpoint}")
        if checkpoint.stat().st_size != int(record["source_checkpoint_bytes"]):
            raise RuntimeError(f"checkpoint byte mismatch for {worker.run_id}")
        if hash_all and sha256_file(checkpoint) != record["source_checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint SHA-256 mismatch for {worker.run_id}")


def prepare_corpus(drive_folder: Path) -> PreparedCorpus:
    archive = verify_archive_identity(drive_folder)
    records = read_archive_manifest(archive)
    extraction_root = Path(drive_folder) / "corpus" / ARCHIVE_SHA256
    corpus_root = extraction_root / ARCHIVE_ROOT
    safe_extract_archive(archive, extraction_root)
    _validate_extracted_checkpoints(corpus_root, records, hash_all=True)
    marker = {
        "schema": CORPUS_SCHEMA,
        "created_at": utc_now(),
        "archive": {
            "name": ARCHIVE_NAME,
            "bytes": ARCHIVE_BYTES,
            "sha256": ARCHIVE_SHA256,
            "root": ARCHIVE_ROOT,
        },
        "corpus_root": str(corpus_root),
        "records": [records[worker.run_id] for worker in WORKERS],
    }
    _atomic_json(_corpus_marker_path(drive_folder), marker)
    return PreparedCorpus(Path(drive_folder), corpus_root, records)


def load_prepared_corpus(drive_folder: Path) -> PreparedCorpus:
    marker_path = _corpus_marker_path(drive_folder)
    if not marker_path.is_file():
        raise RuntimeError(f"checkpoint corpus is not prepared: {marker_path}")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    expected = {
        "name": ARCHIVE_NAME,
        "bytes": ARCHIVE_BYTES,
        "sha256": ARCHIVE_SHA256,
        "root": ARCHIVE_ROOT,
    }
    if marker.get("schema") != CORPUS_SCHEMA or marker.get("archive") != expected:
        raise RuntimeError("prepared Peptides corpus identity does not match this controller")
    records = validate_archive_manifest(marker.get("records", []))
    corpus_root = Path(marker.get("corpus_root", ""))
    _validate_extracted_checkpoints(corpus_root, records, hash_all=False)
    return PreparedCorpus(Path(drive_folder), corpus_root, records)


def ensure_prepared_corpus(
    drive_folder: Path,
    *,
    wait_seconds: int = 1_800,
    reclaim_setup_lock: bool = False,
) -> PreparedCorpus:
    """Prepare once; a concurrently launched second notebook waits for the same corpus."""

    deadline = time.monotonic() + int(wait_seconds)
    lock_path = Path(drive_folder) / "_locks" / "peptides_corpus_setup.lock"
    if reclaim_setup_lock:
        with drive_lock(drive_folder, "peptides_corpus_setup", reclaim=True):
            with contextlib.suppress(Exception):
                return load_prepared_corpus(drive_folder)
            print("[setup] reclaiming stale setup claim and repairing the corpus", flush=True)
            return prepare_corpus(drive_folder)
    while True:
        with contextlib.suppress(Exception):
            return load_prepared_corpus(drive_folder)
        if lock_path.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for corpus setup lock {lock_path}")
            print(
                "[setup] another Peptides notebook is preparing the corpus; waiting...", flush=True
            )
            time.sleep(5)
            continue
        try:
            with drive_lock(drive_folder, "peptides_corpus_setup"):
                with contextlib.suppress(Exception):
                    return load_prepared_corpus(drive_folder)
                print("[setup] verifying and extracting the 30-checkpoint corpus", flush=True)
                return prepare_corpus(drive_folder)
        except RuntimeError:
            if lock_path.exists() and time.monotonic() < deadline:
                time.sleep(5)
                continue
            raise


def auto_graphs_per_batch(
    *,
    device_name: str | None = None,
    total_memory_bytes: int | None = None,
) -> int:
    """Aggressive Peptides batching; exact CUDA OOM backoff remains enabled."""

    if device_name is None or total_memory_bytes is None:
        try:
            import torch

            if not torch.cuda.is_available():
                return 2
            properties = torch.cuda.get_device_properties(0)
            device_name = str(properties.name)
            total_memory_bytes = int(properties.total_memory)
        except (ImportError, RuntimeError):
            return 2
    gib = float(total_memory_bytes) / 1024**3
    name = str(device_name).lower()
    if ("a100" in name or "h100" in name) and gib >= 75:
        return 16
    if gib >= 70:
        return 12
    if gib >= 38:
        return 8
    if gib >= 20:
        return 4
    return 2


def build_production_config(
    drive_folder: Path,
    corpus: PreparedCorpus,
    *,
    graphs_per_batch: int | None = None,
    accelerator: str = "cuda:0",
    strict_audits: bool = False,
) -> MethodologyConfig:
    drive_folder = Path(drive_folder)
    subset = {"train": 2_000, "val": 136, "test": 136, "seed": 31_415}
    task_overrides = {}
    for task in TASKS:
        dataset = "func" if task.startswith("peptides_func_") else "struct"
        task_overrides[task] = {
            "drive_dir": str(drive_folder / "runtime" / task),
            "dataset_dir": str(drive_folder / "datasets" / dataset),
            "grit_repo_dir": f"/content/GRIT_gsm_{task}",
            "analysis_split_limits": dict(subset),
            # Metrics are finite-forward checks on a deterministic analysis subset, not the
            # archive's full-split selection metrics. Strict load/SHA/parameter checks remain.
            "disable_metric_abort_guard": True,
        }
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
            graphs_per_batch=int(graphs_per_batch or auto_graphs_per_batch()),
            oom_backoff=True,
            replica_pair_budget=None,
            jacobian_output_chunk=11,
            progress_heartbeat_seconds=30.0,
        ),
        analysis_seed=31_415,
        accelerator=accelerator,
        num_threads=8,
        checkpoints=corpus.checkpoints,
        task_overrides=task_overrides,
        skip_install=True,
        resume=True,
        force=False,
        strict_audits=bool(strict_audits),
        compute_beneficial_carriage=True,
    )
    config.validate()
    if len(config.checkpoints) != 30:
        raise RuntimeError("Peptides production config must bind all 30 checkpoints")
    return config


def _completion_path(config: MethodologyConfig, worker: WorkerSpec) -> Path:
    return Path(config.output_dir) / worker.task / f"seed_{worker.seed}" / "worker_complete.json"


def verify_selected_checkpoint(corpus: PreparedCorpus, worker: WorkerSpec) -> str:
    checkpoint = corpus.checkpoint_path(worker.run_id)
    expected = str(corpus.records[worker.run_id]["source_checkpoint_sha256"])
    actual = sha256_file(checkpoint)
    if actual != expected:
        raise RuntimeError(f"checkpoint SHA-256 mismatch for {worker.run_id}")
    return expected


def _metric_verification_record(
    config: MethodologyConfig,
    corpus: PreparedCorpus,
    worker: WorkerSpec,
) -> dict[str, Any]:
    """Record finite subset metrics without comparing them to full-split archive metrics."""

    model_path = _completion_path(config, worker).parent / "model.json"
    model = json.loads(model_path.read_text(encoding="utf-8"))
    record = corpus.records[worker.run_id]
    actual_test = float(model["test_metric"])
    actual_validation = float(model["validation_metric"])
    if not math.isfinite(actual_test) or not math.isfinite(actual_validation):
        raise RuntimeError(f"{worker.run_id} produced a non-finite subset verification metric")
    return {
        "scope": "analysis_subset",
        "subset_test": actual_test,
        "subset_validation": actual_validation,
        "archive_full_split_test": float(record["selected_test"]),
        "archive_full_split_validation": float(record["selected_validation"]),
        "selection_metric": str(record["selection_metric"]),
    }


def validate_worker_outputs(
    config: MethodologyConfig,
    corpus: PreparedCorpus,
    worker: WorkerSpec,
    *,
    checkpoint_digest: str,
) -> Mapping[str, Any]:
    from graph_specialisation_metrics.methodology import validate_measurement_worker

    validated = validate_measurement_worker(config, worker.task, worker.seed)
    try:
        if validated["checkpoint_sha256"] != checkpoint_digest:
            raise RuntimeError(f"{worker.key} cache checkpoint differs from its manifest")
        counts = dict(validated["graph_counts"])
        expected_names = {
            "expected",
            "scores_semantic",
            "scores_structural",
            "carriage_semantic",
            "carriage_structural",
        }
        if set(counts) != expected_names or any(int(value) != 48 for value in counts.values()):
            raise RuntimeError(f"{worker.key} cache graph counts are incomplete: {counts}")
        artifacts = {
            stage: {
                key: str(record[key])
                for key in (
                    "path",
                    "file_sha256",
                    "contract_fingerprint",
                    "event_manifest_hash",
                )
            }
            for stage, record in validated["artifacts"].items()
        }
        if set(artifacts) != {"scores", "carriage"}:
            raise RuntimeError(f"{worker.key} artifact index is incomplete")
        return {
            "schema": COMPLETION_SCHEMA,
            "worker": dataclasses.asdict(worker),
            "protocol_fingerprint": config.fingerprint,
            "checkpoint_sha256": checkpoint_digest,
            "graphs": 48,
            "strict_audits": bool(config.strict_audits),
            "phases": list(config.phases),
            "metrics": _metric_verification_record(config, corpus, worker),
            "artifacts": artifacts,
            "audit_findings": len(validated["audit_findings"]),
            "headline_eligible": bool(validated["headline_eligible"]),
            "completed_at": utc_now(),
        }
    finally:
        del validated
        gc.collect()


def _existing_completion(
    config: MethodologyConfig,
    corpus: PreparedCorpus,
    worker: WorkerSpec,
    *,
    checkpoint_digest: str,
) -> Mapping[str, Any] | None:
    marker_path = _completion_path(config, worker)
    if not marker_path.is_file():
        return None
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    expected = {
        "schema": COMPLETION_SCHEMA,
        "worker": dataclasses.asdict(worker),
        "protocol_fingerprint": config.fingerprint,
        "checkpoint_sha256": checkpoint_digest,
        "graphs": 48,
        "strict_audits": bool(config.strict_audits),
        "phases": list(config.phases),
    }
    if any(marker.get(key) != value for key, value in expected.items()):
        return None
    try:
        fresh = validate_worker_outputs(
            config,
            corpus,
            worker,
            checkpoint_digest=checkpoint_digest,
        )
    except Exception as error:  # noqa: BLE001 - invalid cache must resume, not skip
        print(f"[queue] {worker.run_id} needs repair/resume: {error}", flush=True)
        return None
    _atomic_json(marker_path, fresh)
    return fresh


def _restore_worker_protocol(config: MethodologyConfig, worker: WorkerSpec) -> None:
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
    _atomic_json(_completion_path(config, worker).parent / "protocol.json", record)


def run_dataset_queue(
    config: MethodologyConfig,
    corpus: PreparedCorpus,
    dataset: str,
    *,
    train_seed: int | None = None,
    reclaim_worker_index: int = -1,
) -> tuple[Mapping[str, Any], ...]:
    if dataset not in DATASET_WORKERS:
        raise ValueError("DATASET must be 'func' or 'struct'")
    if train_seed is not None and int(train_seed) not in TRAIN_SEEDS:
        raise ValueError("TRAIN_SEED must be 0, 1, or 2")
    workers = (
        DATASET_WORKERS[dataset]
        if train_seed is None
        else SEED_LANE_WORKERS[(dataset, int(train_seed))]
    )
    selected_indices = {worker.index for worker in workers}
    reclaim_worker_index = int(reclaim_worker_index)
    if reclaim_worker_index >= 0 and reclaim_worker_index not in selected_indices:
        raise ValueError(
            f"RECLAIM_WORKER_INDEX={reclaim_worker_index} is not in the "
            f"{dataset}/seed{train_seed} lane"
        )
    results = []
    for position, worker in enumerate(workers, start=1):
        print(
            f"[queue:{dataset}] {position}/{len(workers)} worker {worker.index:02d} "
            f"{worker.task}:seed{worker.seed}",
            flush=True,
        )
        lock_name = f"peptides_production_{worker.task}_seed{worker.seed}"
        reclaim = worker.index == reclaim_worker_index
        with drive_lock(corpus.drive_folder, lock_name, reclaim=reclaim):
            digest = verify_selected_checkpoint(corpus, worker)
            existing = _existing_completion(
                config,
                corpus,
                worker,
                checkpoint_digest=digest,
            )
            if existing is not None:
                print(f"[queue] worker {worker.index:02d} already validated; skipping", flush=True)
                results.append(existing)
                continue
            marker_path = _completion_path(config, worker)
            with contextlib.suppress(FileNotFoundError):
                marker_path.unlink()
            from graph_specialisation_metrics.methodology.runner import run_worker

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
            _restore_worker_protocol(config, worker)
            completion = validate_worker_outputs(
                config,
                corpus,
                worker,
                checkpoint_digest=digest,
            )
            _atomic_json(marker_path, completion)
            results.append(completion)
        release_component_memory()
    return tuple(results)


def completion_status(drive_folder: Path) -> list[dict[str, Any]]:
    root = Path(drive_folder) / "canonical_outputs"
    rows = []
    for worker in WORKERS:
        output = root / worker.task / f"seed_{worker.seed}"
        marker = output / "worker_complete.json"
        current = False
        with contextlib.suppress(OSError, json.JSONDecodeError):
            payload = json.loads(marker.read_text(encoding="utf-8"))
            current = payload.get("schema") == COMPLETION_SCHEMA and payload.get(
                "worker"
            ) == dataclasses.asdict(worker)
        rows.append(
            {
                "index": worker.index,
                "dataset": worker.dataset,
                "task": worker.task,
                "seed": worker.seed,
                "scores": (output / "cache" / "scores" / "raw.pt").is_file(),
                "carriage": (output / "cache" / "carriage" / "fields.pt").is_file(),
                "complete": current,
            }
        )
    return rows


def _print_status(
    rows: Sequence[Mapping[str, Any]],
    *,
    dataset: str | None = None,
    train_seed: int | None = None,
) -> None:
    for loop_dataset in ("func", "struct"):
        selected = [row for row in rows if row["dataset"] == loop_dataset]
        complete = sum(bool(row["complete"]) for row in selected)
        scores = sum(bool(row["scores"]) for row in selected)
        carriage = sum(bool(row["carriage"]) for row in selected)
        print(
            f"[status:{loop_dataset}] complete={complete}/15 scores={scores}/15 "
            f"carriage={carriage}/15",
            flush=True,
        )
    if dataset is not None and train_seed is not None:
        selected = [
            row for row in rows if row["dataset"] == dataset and int(row["seed"]) == int(train_seed)
        ]
        complete = sum(bool(row["complete"]) for row in selected)
        scores = sum(bool(row["scores"]) for row in selected)
        carriage = sum(bool(row["carriage"]) for row in selected)
        print(
            f"[status:{dataset}:seed{train_seed}] complete={complete}/5 "
            f"scores={scores}/5 carriage={carriage}/5",
            flush=True,
        )


def run_frontend(
    *,
    mode: str,
    dataset: str,
    train_seed: int | None = None,
    drive_folder: str | Path = DEFAULT_DRIVE_FOLDER,
    graphs_per_batch: int | None = None,
    accelerator: str = "cuda:0",
    strict_audits: bool = False,
    reclaim_setup_lock: bool = False,
    reclaim_worker_index: int = -1,
) -> Any:
    """Run one five-checkpoint dataset/seed lane, status, setup, or finalization."""

    mode = str(mode).strip().lower()
    dataset = str(dataset).strip().lower()
    if dataset not in DATASET_WORKERS:
        raise ValueError("DATASET must be 'func' or 'struct'")
    if train_seed is not None:
        train_seed = int(train_seed)
        if train_seed not in TRAIN_SEEDS:
            raise ValueError("TRAIN_SEED must be 0, 1, or 2")
    drive_folder = Path(drive_folder)
    if mode == "status":
        rows = completion_status(drive_folder)
        _print_status(rows, dataset=dataset, train_seed=train_seed)
        return rows
    if mode not in {"setup", "run", "finalize"}:
        raise ValueError("MODE must be setup, run, status, or finalize")
    if mode == "run" and train_seed is None:
        raise ValueError("TRAIN_SEED is required in run mode; use one of 0, 1, or 2")

    corpus = ensure_prepared_corpus(
        drive_folder,
        reclaim_setup_lock=bool(reclaim_setup_lock),
    )
    if mode == "setup":
        print(f"[setup:complete] corpus={corpus.corpus_root}", flush=True)
        return {"corpus_root": str(corpus.corpus_root), "archive_sha256": ARCHIVE_SHA256}

    config = build_production_config(
        drive_folder,
        corpus,
        graphs_per_batch=graphs_per_batch,
        accelerator=accelerator,
        strict_audits=strict_audits,
    )
    if mode == "finalize":
        from graph_specialisation_metrics.methodology.runner import finalize_measurement_run

        result = finalize_measurement_run(config)
        _print_status(
            completion_status(drive_folder),
            dataset=dataset,
            train_seed=train_seed,
        )
        return result

    ensure_runtime_dependencies()
    gpu = require_requested_accelerator(config.accelerator)
    print(
        f"[run] dataset={dataset} train_seed={train_seed} workers=5 graphs_per_batch="
        f"{config.execution.graphs_per_batch} gpu={gpu.get('device_name', accelerator)}",
        flush=True,
    )
    results = run_dataset_queue(
        config,
        corpus,
        dataset,
        train_seed=train_seed,
        reclaim_worker_index=reclaim_worker_index,
    )
    _print_status(
        completion_status(drive_folder),
        dataset=dataset,
        train_seed=train_seed,
    )
    return results
