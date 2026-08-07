"""Scores-only Colab workflow for the ZINC dense/1-hop checkpoint trajectory.

The two notebooks beside this module each process one architecture sequentially.  Every epoch has
an isolated canonical cache root, so reruns can validate and skip complete epochs or resume the
first incomplete graph shard.  Plotting is model-free and reads only the consolidated score
caches written to Drive.
"""

from __future__ import annotations

import contextlib
import csv
import dataclasses
import gc
import io
import json
import os
import shutil
import tarfile
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from experiments.methodology.zinc_qm9_canonical_colab_worker import (
    _atomic_json,
    _atomic_text,
    auto_graphs_per_batch,
    drive_lock,
    ensure_runtime_dependencies,
    release_component_memory,
    repository_commit,
    require_requested_accelerator,
    sha256_file,
    utc_now,
)
from graph_specialisation_metrics.methodology.protocol import (
    PROTOCOL_VERSION,
    BootstrapPolicy,
    ExecutionPolicy,
    FamilyPolicy,
    MethodologyConfig,
    RunSizes,
    stable_hash,
)

ARCHIVE_NAME = "zinc_dense_1hop_seed0_trajectory.tar"
ARCHIVE_SIDECAR_NAME = f"{ARCHIVE_NAME}.sha256"
ARCHIVE_SHA256 = "e8b37688bea8abefba11fe972345c1bf539cdb98d55b87fd625fe2f0e4d33634"
ARCHIVE_BYTES = 71_086_080
ARCHIVE_ROOT = "zinc_dense_1hop_seed0_trajectory"
DEFAULT_DRIVE_FOLDER = Path(
    "/content/drive/MyDrive/graph_specialisation_metrics/"
    "multi_seed_models/multiple_checkpoints_zinc"
)
EPOCHS = (10, 100, 250, 500, 1_000, 1_990)
ARCHITECTURE_TASK = {"dense": "zinc", "1hop": "zinc_1hop"}
SCORE_PHASES = ("scores",)
COMPLETION_SCHEMA = "zinc-checkpoint-trajectory-score-v1"
SETUP_SCHEMA = "zinc-checkpoint-trajectory-setup-v1"

EXPECTED_CHECKPOINT_SHA256 = {
    ("dense", 10): "419a1021d9a16b8cd42acd48f90064a134df2194830911e303e4f25ade6b546d",
    ("dense", 100): "b4f483a7a2d2e4d45260094179a05ac8dc5477ccc4d85a24daac718dfa308ef3",
    ("dense", 250): "b5fcf0f94eebf17c29e730d90a1e78bf8d01058ac4bf3b613c73500f4c81b962",
    ("dense", 500): "a0a9733d56fa840547da485cb981f63877d4b6281a9a4f5256a05f029c4f20d0",
    ("dense", 1_000): "241a78098017db4a178a4e7a295721e4e691fd3c7f8ec1a9916936298c7ac364",
    ("dense", 1_990): "5ebe4a9ea6d17c16b49caec418429beb7f9ab76417cdded7b0e8e675368369ac",
    ("1hop", 10): "10970bdb492a6c063f367fde6d3d8f044abbda71a892d01f5ff56ced19e54ef1",
    ("1hop", 100): "ae1ee1144e7233cc789ad9dee7c7029a2d8b3c6d23f1a3e4b5d092797cb5f5eb",
    ("1hop", 250): "c11e45652df5928898dfc2f722c073f15c9a4be39cf5ce266ef6327e566eaa63",
    ("1hop", 500): "b6b42d2ceec3b184ea8112de1f6e73f6b683f483f54a00b480b00f33da9662b4",
    ("1hop", 1_000): "7bc0d8b6fb071e0647405f7e23ee1f57b05549efd794da6fd533ee26e695d6c3",
    ("1hop", 1_990): "dde5f3fd0cb9f39539688831b7a8a9b7888c937b53c0e846bcbe553b285f0dd4",
}


@dataclass(frozen=True)
class TrajectoryCheckpoint:
    architecture: str
    task: str
    seed: int
    epoch: int
    relative_path: str
    sha256: str
    bytes: int

    @property
    def key(self) -> str:
        return f"{self.architecture}:epoch{self.epoch}"


@dataclass(frozen=True)
class PreparedTrajectory:
    drive_folder: Path
    corpus_root: Path
    records: Mapping[tuple[str, int], TrajectoryCheckpoint]
    dataset_root: Path

    def checkpoint_path(self, record: TrajectoryCheckpoint) -> Path:
        return self.corpus_root / record.relative_path


def _read_sidecar(path: Path) -> tuple[str, str]:
    fields = path.read_text(encoding="utf-8").strip().split()
    if len(fields) != 2:
        raise RuntimeError(f"malformed SHA-256 sidecar {path}; expected HASH FILENAME")
    return fields[0].lower(), fields[1].lstrip("*")


def verify_archive_identity(drive_folder: Path) -> Path:
    folder = Path(drive_folder)
    archive = folder / ARCHIVE_NAME
    sidecar = folder / ARCHIVE_SIDECAR_NAME
    if not archive.is_file() or not sidecar.is_file():
        raise FileNotFoundError(
            f"Upload {ARCHIVE_NAME} and {ARCHIVE_SIDECAR_NAME} directly into {folder}"
        )
    digest, filename = _read_sidecar(sidecar)
    if digest != ARCHIVE_SHA256 or filename != ARCHIVE_NAME:
        raise RuntimeError(
            f"sidecar identifies {digest} {filename}; expected {ARCHIVE_SHA256} {ARCHIVE_NAME}"
        )
    if archive.stat().st_size != ARCHIVE_BYTES:
        raise RuntimeError(
            f"archive byte count is {archive.stat().st_size}; expected {ARCHIVE_BYTES}"
        )
    actual = sha256_file(archive)
    if actual != ARCHIVE_SHA256:
        raise RuntimeError(f"archive SHA-256 is {actual}; expected {ARCHIVE_SHA256}")
    return archive


def _manifest_rows(stream: io.TextIOBase) -> tuple[dict[str, str], ...]:
    reader = csv.DictReader(stream, delimiter="\t")
    expected_fields = ["model", "seed", "epoch", "checkpoint", "sha256"]
    if reader.fieldnames != expected_fields:
        raise RuntimeError(
            f"trajectory manifest fields are {reader.fieldnames!r}; expected {expected_fields!r}"
        )
    return tuple(dict(row) for row in reader)


def read_archive_manifest(archive: Path) -> dict[tuple[str, int], TrajectoryCheckpoint]:
    manifest_name = f"{ARCHIVE_ROOT}/manifest.tsv"
    with tarfile.open(archive, "r:") as bundle:
        members = bundle.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            raise RuntimeError("trajectory archive contains duplicate member names")
        for member in members:
            _safe_member_path(Path("."), member.name)
            if not (member.isfile() or member.isdir()):
                raise RuntimeError(f"unsupported trajectory archive member: {member.name}")
        try:
            member = bundle.getmember(manifest_name)
        except KeyError as error:
            raise RuntimeError(f"trajectory archive is missing {manifest_name}") from error
        raw = bundle.extractfile(member)
        if raw is None:
            raise RuntimeError("trajectory manifest is not a regular file")
        rows = _manifest_rows(io.TextIOWrapper(raw, encoding="utf-8"))
        records: dict[tuple[str, int], TrajectoryCheckpoint] = {}
        for row in rows:
            architecture = row["model"]
            epoch = int(row["epoch"])
            key = (architecture, epoch)
            if key in records:
                raise RuntimeError(f"duplicate trajectory checkpoint {key}")
            if architecture not in ARCHITECTURE_TASK or int(row["seed"]) != 0:
                raise RuntimeError(f"unexpected trajectory row: {row}")
            relative = row["checkpoint"]
            expected_relative = f"{architecture}/epoch{epoch}.ckpt"
            expected_sha = EXPECTED_CHECKPOINT_SHA256.get(key)
            if relative != expected_relative or row["sha256"].lower() != expected_sha:
                raise RuntimeError(f"trajectory row differs from the registered corpus: {row}")
            checkpoint_member = bundle.getmember(f"{ARCHIVE_ROOT}/{relative}")
            if not checkpoint_member.isfile() or checkpoint_member.size <= 0:
                raise RuntimeError(f"invalid checkpoint member {checkpoint_member.name}")
            records[key] = TrajectoryCheckpoint(
                architecture=architecture,
                task=ARCHITECTURE_TASK[architecture],
                seed=0,
                epoch=epoch,
                relative_path=relative,
                sha256=expected_sha,
                bytes=int(checkpoint_member.size),
            )
    expected_keys = set(EXPECTED_CHECKPOINT_SHA256)
    if set(records) != expected_keys:
        raise RuntimeError(
            f"trajectory manifest keys differ: missing={sorted(expected_keys - set(records))}, "
            f"extra={sorted(set(records) - expected_keys)}"
        )
    return records


def _safe_member_path(destination: Path, member_name: str) -> Path:
    relative = PurePosixPath(member_name)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise RuntimeError(f"unsafe trajectory archive path: {member_name!r}")
    if relative.parts[0] != ARCHIVE_ROOT:
        raise RuntimeError(f"trajectory member escapes registered root: {member_name!r}")
    return Path(destination).joinpath(*relative.parts)


def _safe_extract_archive(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:") as bundle:
        for member in bundle.getmembers():
            target = _safe_member_path(destination, member.name)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise RuntimeError(f"unsupported trajectory archive member: {member.name}")
            source = bundle.extractfile(member)
            if source is None:
                raise RuntimeError(f"could not read trajectory archive member {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.partial")
            try:
                with temporary.open("xb") as output:
                    shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
                os.replace(temporary, target)
            finally:
                with contextlib.suppress(FileNotFoundError):
                    temporary.unlink()


def _corpus_marker_path(drive_folder: Path) -> Path:
    return Path(drive_folder) / "corpus" / "trajectory_ready.json"


def _validate_checkpoint_files(
    corpus_root: Path,
    records: Mapping[tuple[str, int], TrajectoryCheckpoint],
    *,
    hash_all: bool,
) -> None:
    for record in records.values():
        path = corpus_root / record.relative_path
        if not path.is_file() or path.stat().st_size != record.bytes:
            raise RuntimeError(f"trajectory checkpoint is missing or truncated: {path}")
        if hash_all and sha256_file(path) != record.sha256:
            raise RuntimeError(f"trajectory checkpoint SHA-256 mismatch: {path}")


def prepare_corpus(
    drive_folder: Path,
) -> tuple[Path, Mapping[tuple[str, int], TrajectoryCheckpoint]]:
    archive = verify_archive_identity(drive_folder)
    records = read_archive_manifest(archive)
    extraction_root = Path(drive_folder) / "corpus" / ARCHIVE_SHA256
    corpus_root = extraction_root / ARCHIVE_ROOT
    _safe_extract_archive(archive, extraction_root)
    _validate_checkpoint_files(corpus_root, records, hash_all=True)
    _atomic_json(
        _corpus_marker_path(drive_folder),
        {
            "schema": "zinc-checkpoint-trajectory-corpus-v1",
            "created_at": utc_now(),
            "archive": {
                "name": ARCHIVE_NAME,
                "bytes": ARCHIVE_BYTES,
                "sha256": ARCHIVE_SHA256,
                "root": ARCHIVE_ROOT,
            },
            "corpus_root": str(corpus_root),
            "records": [dataclasses.asdict(records[key]) for key in sorted(records)],
        },
    )
    return corpus_root, records


def load_prepared_corpus(
    drive_folder: Path, *, hash_all: bool = False
) -> tuple[Path, Mapping[tuple[str, int], TrajectoryCheckpoint]]:
    marker_path = _corpus_marker_path(drive_folder)
    if not marker_path.is_file():
        raise RuntimeError(f"trajectory corpus marker is missing: {marker_path}")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    expected_archive = {
        "name": ARCHIVE_NAME,
        "bytes": ARCHIVE_BYTES,
        "sha256": ARCHIVE_SHA256,
        "root": ARCHIVE_ROOT,
    }
    if (
        marker.get("schema") != "zinc-checkpoint-trajectory-corpus-v1"
        or marker.get("archive") != expected_archive
    ):
        raise RuntimeError("prepared trajectory corpus has the wrong identity")
    records = {
        (str(row["architecture"]), int(row["epoch"])): TrajectoryCheckpoint(**row)
        for row in marker.get("records", ())
    }
    if {key: record.sha256 for key, record in records.items()} != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("prepared trajectory checkpoint manifest is not registered")
    corpus_root = Path(marker.get("corpus_root", ""))
    _validate_checkpoint_files(corpus_root, records, hash_all=hash_all)
    return corpus_root, records


def _file_inventory(root: Path) -> list[dict[str, Any]]:
    return [
        {"path": str(path.relative_to(root)), "bytes": int(path.stat().st_size)}
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]


def _validate_inventory(root: Path, records: Sequence[Mapping[str, Any]]) -> None:
    if not records:
        raise RuntimeError(f"dataset inventory is empty for {root}")
    for record in records:
        path = root / str(record["path"])
        if not path.is_file() or path.stat().st_size != int(record["bytes"]):
            raise RuntimeError(f"dataset file is missing or truncated: {path}")


def _trajectory_dataset_marker(drive_folder: Path) -> Path:
    return Path(drive_folder) / "zinc_dataset_ready.json"


def load_zinc_dataset(drive_folder: Path) -> Path:
    """Reuse the parent canonical ZINC cache, or a trajectory-specific warmup marker."""

    parent = Path(drive_folder).parent
    shared_marker = parent / "datasets_ready.json"
    if shared_marker.is_file():
        marker = json.loads(shared_marker.read_text(encoding="utf-8"))
        root = Path(marker["roots"]["zinc"])
        _validate_inventory(root, marker["files"]["zinc"])
        return root
    marker_path = _trajectory_dataset_marker(drive_folder)
    if not marker_path.is_file():
        raise RuntimeError("no prepared ZINC dataset marker is available")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("schema") != "zinc-checkpoint-trajectory-dataset-v1":
        raise RuntimeError("trajectory ZINC dataset marker has the wrong schema")
    root = Path(marker["root"])
    _validate_inventory(root, marker["files"])
    return root


def prepare_zinc_dataset(drive_folder: Path) -> Path:
    try:
        return load_zinc_dataset(drive_folder)
    except (OSError, KeyError, RuntimeError, json.JSONDecodeError):
        pass
    from torch_geometric.datasets import ZINC

    root = Path(drive_folder).parent / "datasets" / "zinc"
    sizes = {
        split: len(ZINC(str(root), subset=True, split=split)) for split in ("train", "val", "test")
    }
    inventory = _file_inventory(root)
    _validate_inventory(root, inventory)
    _atomic_json(
        _trajectory_dataset_marker(drive_folder),
        {
            "schema": "zinc-checkpoint-trajectory-dataset-v1",
            "created_at": utc_now(),
            "root": str(root),
            "sizes": sizes,
            "files": inventory,
        },
    )
    return root


def _setup_marker_path(drive_folder: Path) -> Path:
    return Path(drive_folder) / "trajectory_setup_ready.json"


def load_prepared_trajectory(drive_folder: Path, *, hash_all: bool = False) -> PreparedTrajectory:
    marker_path = _setup_marker_path(drive_folder)
    if not marker_path.is_file():
        raise RuntimeError(f"trajectory setup marker is missing: {marker_path}")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("schema") != SETUP_SCHEMA or marker.get("archive_sha256") != ARCHIVE_SHA256:
        raise RuntimeError("trajectory setup marker has the wrong identity")
    corpus_root, records = load_prepared_corpus(drive_folder, hash_all=hash_all)
    dataset_root = load_zinc_dataset(drive_folder)
    return PreparedTrajectory(Path(drive_folder), corpus_root, records, dataset_root)


def _prepare_setup_claimed(drive_folder: Path) -> PreparedTrajectory:
    try:
        prepared = load_prepared_trajectory(drive_folder, hash_all=True)
        verify_archive_identity(drive_folder)
        print("[setup] reusing validated trajectory corpus and ZINC dataset", flush=True)
        return prepared
    except (OSError, KeyError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"[setup] preparing or repairing trajectory assets ({error})", flush=True)
    corpus_root, records = prepare_corpus(drive_folder)
    dataset_root = prepare_zinc_dataset(drive_folder)
    _atomic_json(
        _setup_marker_path(drive_folder),
        {
            "schema": SETUP_SCHEMA,
            "created_at": utc_now(),
            "repository_commit": repository_commit(),
            "archive_sha256": ARCHIVE_SHA256,
            "corpus_root": str(corpus_root),
            "dataset_root": str(dataset_root),
            "checkpoints": len(records),
        },
    )
    return PreparedTrajectory(Path(drive_folder), corpus_root, records, dataset_root)


def setup_drive(
    drive_folder: Path,
    *,
    reclaim_stale_lock: bool = False,
    wait_seconds: int = 1_200,
) -> PreparedTrajectory:
    """Prepare once; a parallel notebook waits for the first setup claim to finish."""

    folder = Path(drive_folder)
    folder.mkdir(parents=True, exist_ok=True)
    try:
        return load_prepared_trajectory(folder)
    except (OSError, KeyError, RuntimeError, ValueError, json.JSONDecodeError):
        pass
    try:
        with drive_lock(folder, "trajectory_setup", reclaim=reclaim_stale_lock):
            return _prepare_setup_claimed(folder)
    except RuntimeError as error:
        if "Drive claim already exists" not in str(error) or reclaim_stale_lock:
            raise
        print(
            "[setup] the other trajectory notebook is preparing shared assets; waiting", flush=True
        )
        deadline = time.monotonic() + int(wait_seconds)
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            time.sleep(5)
            try:
                prepared = load_prepared_trajectory(folder)
                print("[setup] shared trajectory assets are ready", flush=True)
                return prepared
            except (OSError, KeyError, RuntimeError, ValueError, json.JSONDecodeError) as pending:
                last_error = pending
        raise RuntimeError(
            "timed out waiting for trajectory setup; if the other runtime died, rerun one "
            "notebook with RECLAIM_SETUP_LOCK=True"
        ) from last_error


def architecture_records(
    prepared: PreparedTrajectory, architecture: str
) -> tuple[TrajectoryCheckpoint, ...]:
    if architecture not in ARCHITECTURE_TASK:
        raise ValueError(
            f"architecture must be one of {tuple(ARCHITECTURE_TASK)}, got {architecture!r}"
        )
    return tuple(prepared.records[(architecture, epoch)] for epoch in EPOCHS)


def epoch_output_dir(drive_folder: Path, architecture: str, epoch: int) -> Path:
    return (
        Path(drive_folder) / "score_trajectory_outputs" / architecture / f"epoch_{int(epoch):04d}"
    )


def build_epoch_config(
    prepared: PreparedTrajectory,
    record: TrajectoryCheckpoint,
    *,
    graphs_per_batch: int | None = None,
    accelerator: str = "cuda:0",
    strict_audits: bool = False,
) -> MethodologyConfig:
    checkpoint = prepared.checkpoint_path(record)
    batch = int(graphs_per_batch or auto_graphs_per_batch())
    config = MethodologyConfig(
        output_dir=str(epoch_output_dir(prepared.drive_folder, record.architecture, record.epoch)),
        tasks=(record.task,),
        train_seeds=(record.seed,),
        phases=SCORE_PHASES,
        sizes=RunSizes(),
        bootstrap=BootstrapPolicy(rng_seed=17_071, replicates=2_000, resample_source=True),
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
        checkpoints={f"{record.task}:{record.seed}": str(checkpoint)},
        task_overrides={
            record.task: {
                "drive_dir": str(prepared.drive_folder / "runtime" / record.architecture),
                "dataset_dir": str(prepared.dataset_root),
            }
        },
        skip_install=True,
        resume=True,
        force=False,
        strict_audits=bool(strict_audits),
    )
    config.validate()
    return config


def _required_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"required score sidecar is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"score sidecar is not a JSON object: {path}")
    return value


def _exact_graph_ids(value: Any, *, label: str) -> tuple[set[int], Mapping[Any, Any]]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} is not a graph-keyed mapping")
    try:
        graph_ids = {int(key) for key in value}
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{label} contains a non-integer graph ID") from error
    if len(graph_ids) != len(value):
        raise RuntimeError(f"{label} contains duplicate integer graph IDs")
    return graph_ids, value


def validate_score_output(
    config: MethodologyConfig,
    record: TrajectoryCheckpoint,
    *,
    checkpoint_digest: str | None = None,
) -> Mapping[str, Any]:
    """Model-free, fail-closed validation of one scores-only epoch cache."""

    from graph_specialisation_metrics.methodology.cache import load_cache_artifact_file
    from graph_specialisation_metrics.methodology.tasks import get_task

    root = config.root / record.task / f"seed_{record.seed}"
    partials = (
        sorted(path for path in root.rglob("*.partial") if path.is_file()) if root.exists() else []
    )
    if partials:
        raise RuntimeError(f"unfinished score-cache writes remain: {partials}")
    protocol = _required_json(root / "protocol.json")
    expected_protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "fingerprint": config.fingerprint,
        "execution_mode": "isolated-seed-worker",
        "worker_task": record.task,
        "worker_seed": record.seed,
        "phases": list(SCORE_PHASES),
    }
    if any(protocol.get(key) != value for key, value in expected_protocol.items()):
        raise RuntimeError(f"worker protocol does not match epoch {record.epoch}")
    model = _required_json(root / "model.json")
    if (
        model.get("protocol_version") != PROTOCOL_VERSION
        or model.get("task") != record.task
        or int(model.get("train_seed", -1)) != record.seed
    ):
        raise RuntimeError("model sidecar has the wrong protocol/task/seed")
    configured_task = get_task(record.task)
    expected_adapter = configured_task.adapter_version
    if model.get("task_adapter_version") != expected_adapter:
        raise RuntimeError("model sidecar uses a stale task adapter")
    geometry = model.get("model_geometry")
    if not isinstance(geometry, Mapping) or not {
        "layers",
        "heads",
        "head_width",
        "hidden_width",
        "outputs",
    } <= set(geometry):
        raise RuntimeError("model sidecar has incomplete geometry")
    if any(
        int(geometry[name]) <= 0
        for name in ("layers", "heads", "head_width", "hidden_width", "outputs")
    ):
        raise RuntimeError("model sidecar has non-positive geometry")
    expected_sigma = configured_task.output.resolve(int(geometry["outputs"]))
    if model.get(
        "output_representation"
    ) != configured_task.output.representation or not np.array_equal(
        np.asarray(model.get("sigma"), dtype=np.float64), expected_sigma
    ):
        raise RuntimeError("model sidecar uses a stale output representation or scale")
    if int(model.get("parameter_count", 0)) <= 0:
        raise RuntimeError("model sidecar has no parameter count")
    canonical_audits = model.get("canonical_audits")
    model_failures = (
        canonical_audits.get("failures") if isinstance(canonical_audits, Mapping) else None
    )
    if (
        not isinstance(canonical_audits, Mapping)
        or isinstance(model_failures, (str, bytes))
        or not isinstance(model_failures, Sequence)
    ):
        raise TypeError("model sidecar has no canonical audit record")
    metrics = np.asarray([model.get("validation_metric"), model.get("test_metric")], dtype=float)
    if not np.isfinite(metrics).all():
        raise RuntimeError("model sidecar has non-finite validation/test metrics")
    splits = model.get("splits")
    if not isinstance(splits, Mapping) or not isinstance(splits.get("discovery"), Sequence):
        raise TypeError("model sidecar has no discovery split")
    expected_graph_ids = {int(value) for value in splits["discovery"]}
    if len(expected_graph_ids) != config.sizes.discovery_graphs:
        raise RuntimeError("model discovery split has the wrong size")
    audits = _required_json(root / "audits.json")
    findings = audits.get("findings")
    if (
        audits.get("protocol_version") != PROTOCOL_VERSION
        or audits.get("task") != record.task
        or int(audits.get("train_seed", -1)) != record.seed
        or audits.get("phases") != list(SCORE_PHASES)
        or bool(audits.get("strict_audits")) != bool(config.strict_audits)
        or isinstance(findings, (str, bytes))
        or not isinstance(findings, Sequence)
        or bool(audits.get("headline_eligible")) != (not bool(findings))
    ):
        raise RuntimeError("worker audit sidecar is incomplete or inconsistent")
    if config.strict_audits and findings:
        raise RuntimeError("strict-audit score worker contains audit findings")

    artifact = load_cache_artifact_file(root / "cache" / "scores" / "raw.pt")
    try:
        contract = artifact.metadata["contract"]
        digest = checkpoint_digest or sha256_file(Path(config.checkpoints[f"{record.task}:0"]))
        expected_contract = {
            "protocol_fingerprint": config.fingerprint,
            "task": record.task,
            "task_adapter_version": expected_adapter,
            "checkpoint_sha256": record.sha256,
            "train_seed": record.seed,
            "donors_per_source": config.sizes.donors_per_source,
            "source_cap": config.sizes.sources_per_graph,
            "bootstrap_seed": config.bootstrap.rng_seed,
            "bootstrap_replicates": config.bootstrap.replicates,
        }
        if digest != record.sha256 or any(
            contract.get(key) != value for key, value in expected_contract.items()
        ):
            raise RuntimeError("score cache contract or checkpoint digest differs")
        if contract.get("model_geometry") != geometry:
            raise RuntimeError("score cache/model geometry differs")
        if contract.get("output_representation") != model.get("output_representation"):
            raise RuntimeError("score cache/model output representation differs")
        if not np.array_equal(
            np.asarray(contract.get("sigma"), dtype=np.float64),
            np.asarray(model.get("sigma"), dtype=np.float64),
        ):
            raise RuntimeError("score cache/model output scale differs")
        if contract.get("split_fingerprint") != stable_hash(dict(splits)):
            raise RuntimeError("score cache/model split fingerprint differs")
        if model.get("checkpoint_sha256") != record.sha256:
            raise RuntimeError("model sidecar has the wrong checkpoint SHA-256")
        payload = artifact.value
        if not isinstance(payload, Mapping) or payload.get("protocol_version") != PROTOCOL_VERSION:
            raise RuntimeError("score cache payload is malformed")
        if payload.get("manifest_hash") != contract.get("event_manifest_hash"):
            raise RuntimeError("score cache event manifest is inconsistent")
        channels = payload.get("channels")
        if not isinstance(channels, Mapping) or set(channels) != {"semantic", "structural"}:
            raise RuntimeError("score cache does not contain exactly both channels")
        raw_by_channel: dict[str, np.ndarray] = {}
        for channel in ("semantic", "structural"):
            channel_payload = channels[channel]
            graph_ids, graph_scores = _exact_graph_ids(
                channel_payload.get("graph_scores"), label=f"{channel} graph_scores"
            )
            if graph_ids != expected_graph_ids:
                raise RuntimeError(f"{channel} score cache has the wrong graph IDs")
            raw = np.asarray(channel_payload.get("raw"), dtype=np.float64)
            expected_shape = (int(geometry["layers"]), int(geometry["heads"]))
            if raw.shape != expected_shape or not np.isfinite(raw).all():
                raise RuntimeError(f"{channel} raw score matrix is malformed")
            for graph_id, values in graph_scores.items():
                array = np.asarray(values, dtype=np.float64)
                if array.shape != expected_shape or not np.isfinite(array).all():
                    raise RuntimeError(f"{channel} graph {graph_id} score matrix is malformed")
            raw_by_channel[channel] = raw.copy()
        return {
            "architecture": record.architecture,
            "task": record.task,
            "seed": record.seed,
            "epoch": record.epoch,
            "checkpoint_sha256": record.sha256,
            "protocol_fingerprint": config.fingerprint,
            "split_fingerprint": str(contract["split_fingerprint"]),
            "event_manifest_hash": str(contract["event_manifest_hash"]),
            "artifact_path": str(artifact.path),
            "artifact_sha256": artifact.file_sha256,
            "contract_fingerprint": str(artifact.metadata["contract_fingerprint"]),
            "raw": raw_by_channel,
            "model_geometry": dict(geometry),
            "validation_metric": float(model["validation_metric"]),
            "test_metric": float(model["test_metric"]),
            "audit_findings": len(findings),
            "headline_eligible": bool(audits["headline_eligible"]),
        }
    finally:
        del artifact
        gc.collect()


def _completion_path(config: MethodologyConfig, record: TrajectoryCheckpoint) -> Path:
    return config.root / record.task / f"seed_{record.seed}" / "trajectory_score_complete.json"


def _completion_record(validated: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": COMPLETION_SCHEMA,
        "architecture": validated["architecture"],
        "task": validated["task"],
        "seed": validated["seed"],
        "epoch": validated["epoch"],
        "checkpoint_sha256": validated["checkpoint_sha256"],
        "protocol_fingerprint": validated["protocol_fingerprint"],
        "split_fingerprint": validated["split_fingerprint"],
        "event_manifest_hash": validated["event_manifest_hash"],
        "artifact_path": validated["artifact_path"],
        "artifact_sha256": validated["artifact_sha256"],
        "contract_fingerprint": validated["contract_fingerprint"],
        "audit_findings": validated["audit_findings"],
        "headline_eligible": validated["headline_eligible"],
        "completed_at": utc_now(),
    }


def run_architecture(
    prepared: PreparedTrajectory,
    architecture: str,
    *,
    graphs_per_batch: int | None = None,
    accelerator: str = "cuda:0",
    strict_audits: bool = False,
    reclaim_epoch: int = -1,
) -> Mapping[str, Any]:
    """Run six checkpoints sequentially, validating before every skip."""

    from graph_specialisation_metrics.methodology.runner import run_worker

    records = architecture_records(prepared, architecture)
    if int(reclaim_epoch) >= 0 and int(reclaim_epoch) not in EPOCHS:
        raise ValueError(f"RECLAIM_EPOCH must be -1 or one of {EPOCHS}")
    completed: list[Mapping[str, Any]] = []
    skipped: list[int] = []
    for position, record in enumerate(records, start=1):
        print(
            f"[trajectory] {architecture} {position}/{len(records)}: epoch {record.epoch}",
            flush=True,
        )
        config = build_epoch_config(
            prepared,
            record,
            graphs_per_batch=graphs_per_batch,
            accelerator=accelerator,
            strict_audits=strict_audits,
        )
        claim = f"trajectory_{architecture}_epoch{record.epoch}"
        with drive_lock(
            prepared.drive_folder,
            claim,
            reclaim=(int(reclaim_epoch) == record.epoch),
        ):
            checkpoint = prepared.checkpoint_path(record)
            digest = sha256_file(checkpoint)
            if digest != record.sha256:
                raise RuntimeError(f"checkpoint SHA-256 mismatch for {record.key}")
            try:
                validated = validate_score_output(config, record, checkpoint_digest=digest)
            except Exception as error:  # noqa: BLE001 - a partial epoch is resumed below
                print(f"[trajectory] epoch {record.epoch} needs run/resume ({error})", flush=True)
            else:
                _atomic_json(_completion_path(config, record), _completion_record(validated))
                print(f"[trajectory] epoch {record.epoch} already validated; skipping", flush=True)
                completed.append(_completion_record(validated))
                skipped.append(record.epoch)
                del validated
                gc.collect()
                continue
            with contextlib.suppress(FileNotFoundError):
                _completion_path(config, record).unlink()
            result = None
            try:
                result = run_worker(
                    config,
                    record.task,
                    record.seed,
                    retain_results=False,
                )
            finally:
                if result is not None:
                    del result
                release_component_memory()
            validated = validate_score_output(config, record, checkpoint_digest=digest)
            marker = _completion_record(validated)
            _atomic_json(_completion_path(config, record), marker)
            completed.append(marker)
            del validated
            gc.collect()
        print(f"[trajectory] epoch {record.epoch} complete", flush=True)
    return {
        "mode": "run",
        "architecture": architecture,
        "epochs": list(EPOCHS),
        "skipped_epochs": skipped,
        "completed": completed,
        "finished_at": utc_now(),
    }


def _atomic_figure(figure: Any, path: Path, *, format_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        figure.savefig(temporary, format=format_name, dpi=220, bbox_inches="tight")
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def plot_architecture(
    prepared: PreparedTrajectory,
    architecture: str,
    *,
    graphs_per_batch: int | None = None,
    accelerator: str = "cuda:0",
    strict_audits: bool = False,
) -> Mapping[str, Any]:
    """Validate all six score caches and plot per-head distributions over epoch."""

    import matplotlib.pyplot as plt

    validated_rows: list[Mapping[str, Any]] = []
    for record in architecture_records(prepared, architecture):
        config = build_epoch_config(
            prepared,
            record,
            graphs_per_batch=graphs_per_batch,
            accelerator=accelerator,
            strict_audits=strict_audits,
        )
        validated_rows.append(validate_score_output(config, record))
    protocol_fingerprints = {row["protocol_fingerprint"] for row in validated_rows}
    split_fingerprints = {row["split_fingerprint"] for row in validated_rows}
    geometries = {json.dumps(row["model_geometry"], sort_keys=True) for row in validated_rows}
    if len(protocol_fingerprints) != 1 or len(split_fingerprints) != 1 or len(geometries) != 1:
        raise RuntimeError("trajectory epochs do not share one scientific contract/split/geometry")

    output_dir = Path(prepared.drive_folder) / "score_trajectory_plots" / architecture
    output_dir.mkdir(parents=True, exist_ok=True)
    long_buffer = io.StringIO()
    long_writer = csv.writer(long_buffer)
    long_writer.writerow(["architecture", "epoch", "channel", "layer", "head", "score"])
    summary_buffer = io.StringIO()
    summary_writer = csv.writer(summary_buffer)
    summary_writer.writerow(
        [
            "architecture",
            "epoch",
            "channel",
            "validation_metric",
            "test_metric",
            "count",
            "mean",
            "std",
            "min",
            "q25",
            "median",
            "q75",
            "max",
        ]
    )
    for row in validated_rows:
        for channel in ("semantic", "structural"):
            raw = np.asarray(row["raw"][channel], dtype=np.float64)
            for layer, head in np.ndindex(raw.shape):
                long_writer.writerow(
                    [architecture, row["epoch"], channel, layer, head, f"{raw[layer, head]:.17g}"]
                )
            values = raw.reshape(-1)
            q25, median, q75 = np.quantile(values, [0.25, 0.5, 0.75])
            summary_writer.writerow(
                [
                    architecture,
                    row["epoch"],
                    channel,
                    f"{row['validation_metric']:.17g}",
                    f"{row['test_metric']:.17g}",
                    values.size,
                    f"{values.mean():.17g}",
                    f"{values.std():.17g}",
                    f"{values.min():.17g}",
                    f"{q25:.17g}",
                    f"{median:.17g}",
                    f"{q75:.17g}",
                    f"{values.max():.17g}",
                ]
            )
    long_path = output_dir / "head_scores_long.csv"
    summary_path = output_dir / "score_summary.csv"
    _atomic_text(long_path, long_buffer.getvalue())
    _atomic_text(summary_path, summary_buffer.getvalue())

    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharex=True)
    positions = np.arange(len(validated_rows), dtype=float)
    for axis, channel in zip(axes, ("semantic", "structural"), strict=True):
        distributions = [np.asarray(row["raw"][channel]).reshape(-1) for row in validated_rows]
        violins = axis.violinplot(
            distributions,
            positions=positions,
            widths=0.78,
            showmeans=False,
            showmedians=True,
            showextrema=True,
        )
        for body in violins["bodies"]:
            body.set_facecolor("#4472C4" if channel == "semantic" else "#ED7D31")
            body.set_edgecolor("#202020")
            body.set_alpha(0.72)
        medians = [float(np.median(values)) for values in distributions]
        axis.plot(positions, medians, color="#111111", marker="o", linewidth=1.2, label="median")
        positive = all(np.all(values > 0) for values in distributions)
        if positive:
            axis.set_yscale("log")
        axis.set_title(f"{channel.capitalize()} scores")
        axis.set_xticks(positions, [str(row["epoch"]) for row in validated_rows])
        axis.set_xlabel("Training epoch")
        axis.set_ylabel("Canonical raw head score" + (" (log scale)" if positive else ""))
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False, loc="best")
    label = "Dense" if architecture == "dense" else "1-hop"
    figure.suptitle(f"ZINC {label}: head-score distributions across checkpoints", y=1.02)
    figure.tight_layout()
    png_path = output_dir / "score_distributions.png"
    pdf_path = output_dir / "score_distributions.pdf"
    _atomic_figure(figure, png_path, format_name="png")
    _atomic_figure(figure, pdf_path, format_name="pdf")
    plt.close(figure)
    manifest = {
        "schema": "zinc-checkpoint-trajectory-plot-v1",
        "created_at": utc_now(),
        "repository_commit": repository_commit(),
        "archive_sha256": ARCHIVE_SHA256,
        "architecture": architecture,
        "task": ARCHITECTURE_TASK[architecture],
        "epochs": list(EPOCHS),
        "protocol_fingerprint": next(iter(protocol_fingerprints)),
        "split_fingerprint": next(iter(split_fingerprints)),
        "checkpoints": {str(row["epoch"]): row["checkpoint_sha256"] for row in validated_rows},
        "checkpoint_metrics": {
            str(row["epoch"]): {
                "validation": row["validation_metric"],
                "test": row["test_metric"],
            }
            for row in validated_rows
        },
        "score_artifacts": {
            str(row["epoch"]): {
                "path": row["artifact_path"],
                "sha256": row["artifact_sha256"],
                "contract_fingerprint": row["contract_fingerprint"],
                "event_manifest_hash": row["event_manifest_hash"],
            }
            for row in validated_rows
        },
        "files": {
            "png": str(png_path),
            "pdf": str(pdf_path),
            "long_csv": str(long_path),
            "summary_csv": str(summary_path),
        },
    }
    manifest_path = output_dir / "plot_manifest.json"
    _atomic_json(manifest_path, manifest)
    for row in validated_rows:
        row["raw"].clear()
    del validated_rows
    gc.collect()
    print(f"[plot] wrote {png_path}", flush=True)
    return {**manifest, "manifest": str(manifest_path)}


def completion_status(drive_folder: Path, architecture: str) -> list[dict[str, Any]]:
    if architecture not in ARCHITECTURE_TASK:
        raise ValueError(f"unknown architecture {architecture!r}")
    task = ARCHITECTURE_TASK[architecture]
    rows = []
    for epoch in EPOCHS:
        root = epoch_output_dir(drive_folder, architecture, epoch) / task / "seed_0"
        rows.append(
            {
                "architecture": architecture,
                "epoch": epoch,
                "scores": (root / "cache" / "scores" / "raw.pt").is_file(),
                "validated": (root / "trajectory_score_complete.json").is_file(),
            }
        )
    return rows


def run_frontend(
    *,
    mode: str,
    architecture: str,
    drive_folder: str | Path = DEFAULT_DRIVE_FOLDER,
    graphs_per_batch: int | None = None,
    accelerator: str = "cuda:0",
    strict_audits: bool = False,
    reclaim_setup_lock: bool = False,
    reclaim_epoch: int = -1,
) -> Mapping[str, Any] | list[dict[str, Any]]:
    """Notebook entry point: setup, run+plot, plot-only, or lightweight status."""

    mode = str(mode).strip().lower()
    architecture = str(architecture).strip().lower()
    folder = Path(drive_folder)
    if architecture not in ARCHITECTURE_TASK:
        raise ValueError(f"architecture must be one of {tuple(ARCHITECTURE_TASK)}")
    if mode == "status":
        rows = completion_status(folder, architecture)
        for row in rows:
            print(row, flush=True)
        return rows
    if mode not in {"setup", "run", "plot"}:
        raise ValueError("MODE must be setup, run, plot, or status")
    ensure_runtime_dependencies()
    prepared = setup_drive(folder, reclaim_stale_lock=bool(reclaim_setup_lock))
    if mode == "setup":
        return {
            "mode": "setup",
            "drive_folder": str(folder),
            "corpus_root": str(prepared.corpus_root),
            "dataset_root": str(prepared.dataset_root),
            "checkpoints": len(prepared.records),
        }
    if mode == "run":
        require_requested_accelerator(accelerator)
        run_result = run_architecture(
            prepared,
            architecture,
            graphs_per_batch=graphs_per_batch,
            accelerator=accelerator,
            strict_audits=strict_audits,
            reclaim_epoch=reclaim_epoch,
        )
        plot_result = plot_architecture(
            prepared,
            architecture,
            graphs_per_batch=graphs_per_batch,
            accelerator=accelerator,
            strict_audits=strict_audits,
        )
        return {"run": run_result, "plot": plot_result}
    return plot_architecture(
        prepared,
        architecture,
        graphs_per_batch=graphs_per_batch,
        accelerator=accelerator,
        strict_audits=strict_audits,
    )


__all__ = [
    "ARCHITECTURE_TASK",
    "ARCHIVE_SHA256",
    "DEFAULT_DRIVE_FOLDER",
    "EPOCHS",
    "EXPECTED_CHECKPOINT_SHA256",
    "PreparedTrajectory",
    "TrajectoryCheckpoint",
    "architecture_records",
    "build_epoch_config",
    "completion_status",
    "load_prepared_trajectory",
    "plot_architecture",
    "read_archive_manifest",
    "run_architecture",
    "run_frontend",
    "setup_drive",
    "validate_score_output",
    "verify_archive_identity",
]
