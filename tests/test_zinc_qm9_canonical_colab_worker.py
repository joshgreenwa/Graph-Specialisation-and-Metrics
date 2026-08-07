"""Contract tests for the emergency 30-checkpoint Colab frontend."""

from __future__ import annotations

import contextlib
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.methodology import zinc_qm9_canonical_colab_worker as worker  # noqa: I001


NOTEBOOK = ROOT / "experiments" / "methodology" / "zinc_qm9_canonical_worker_colab.ipynb"


def _manifest_records():
    records = []
    for spec in worker.WORKERS:
        records.append(
            {
                "run_id": spec.run_id,
                "archive_checkpoint": (f"checkpoints/{spec.run_id}/best_available.ckpt"),
                "source_checkpoint_sha256": f"{spec.index + 1:064x}",
                "source_checkpoint_bytes": 1_000 + spec.index,
                "exact_global_best": spec.index < worker.EXACT_GLOBAL_BEST,
                "selected_epoch": 100 + spec.index,
                "selected_val_mae": 0.01 + spec.index / 10_000,
                "selected_test_mae": 0.02 + spec.index / 10_000,
            }
        )
    return records


def _corpus(tmp_path: Path) -> worker.PreparedCorpus:
    records = worker.validate_archive_manifest(_manifest_records())
    return worker.PreparedCorpus(
        drive_folder=tmp_path,
        corpus_root=tmp_path / "corpus" / worker.ARCHIVE_ROOT,
        records=records,
    )


def test_fixed_replacement_corpus_identity_and_complete_worker_grid():
    assert worker.ARCHIVE_NAME == "zinc_qm9_best_checkpoints.tar"
    assert worker.ARCHIVE_SIDECAR_NAME == "zinc_qm9_best_checkpoints.tar.sha256"
    assert worker.ARCHIVE_BYTES == 177_950_720
    assert (
        worker.ARCHIVE_SHA256 == "1d41d10b5e5c32fa1ce70535de646c762ba3ad4ad3393406c12b64c21cfced22"
    )
    assert worker.ARCHIVE_ROOT == "zinc_qm9_best_available_20260807_151904"
    assert worker.EXACT_GLOBAL_BEST == 16
    assert worker.BEST_AVAILABLE == 14
    assert worker.DEFAULT_DRIVE_FOLDER == Path(
        "/content/drive/MyDrive/graph_specialisation_metrics/multi_seed_models"
    )
    assert (
        inspect.signature(worker.run_frontend).parameters["drive_folder"].default
        == worker.DEFAULT_DRIVE_FOLDER
    )

    assert len(worker.TASKS) == 10
    assert worker.TRAIN_SEEDS == (0, 1, 2)
    assert len(worker.WORKERS) == 30
    assert [spec.index for spec in worker.WORKERS] == list(range(30))
    assert len({spec.key for spec in worker.WORKERS}) == 30
    assert len({spec.run_id for spec in worker.WORKERS}) == 30
    assert {(spec.task, spec.seed) for spec in worker.WORKERS} == {
        (task, seed) for task in worker.TASKS for seed in worker.TRAIN_SEEDS
    }


def test_manifest_is_mapped_by_explicit_run_id_and_parses_checkpoint_sha():
    records = worker.validate_archive_manifest(_manifest_records())

    assert list(records) == [spec.run_id for spec in worker.WORKERS]
    assert records["zinc.dense.s0"]["source_checkpoint_sha256"] == f"{1:064x}"
    assert (
        records["qm9_gap.2hop_vnode.s2"]["archive_checkpoint"]
        == "checkpoints/qm9_gap.2hop_vnode.s2/best_available.ckpt"
    )

    malformed = _manifest_records()
    malformed[0] = {**malformed[0], "run_id": "zinc.not-a-model.s0"}
    with pytest.raises(RuntimeError, match="unexpected run_id"):
        worker.validate_archive_manifest(malformed)


def test_worker_selector_accepts_index_or_explicit_task_seed():
    by_index = worker.resolve_worker(17)
    by_key = worker.resolve_worker(task=by_index.task, seed=by_index.seed)
    assert by_index == by_key

    with pytest.raises(ValueError, match=r"\[0, 29\]"):
        worker.resolve_worker(30)
    with pytest.raises(ValueError, match="requires both"):
        worker.resolve_worker(task="zinc")


def test_four_queue_schedules_are_exact_disjoint_and_cover_every_worker():
    expected = (
        (0, 4, 8, 12, 16, 20, 24, 28),
        (1, 5, 9, 13, 17, 21, 25, 29),
        (2, 6, 10, 14, 18, 22, 26),
        (3, 7, 11, 15, 19, 23, 27),
    )

    assert worker.QUEUE_LANES == 4
    assert worker.QUEUE_WORKER_INDICES == expected
    assert [len(indices) for indices in expected] == [8, 8, 7, 7]
    flattened = [index for indices in expected for index in indices]
    assert sorted(flattened) == list(range(30))
    assert len(flattened) == len(set(flattened))


def test_worker_queue_selector_parses_commas_spaces_and_rejects_invalid_values():
    selected = worker.resolve_worker_queue("0, 4  8,12")
    assert tuple(spec.index for spec in selected) == (0, 4, 8, 12)
    assert tuple(spec.index for spec in worker.resolve_worker_queue([1, "5"])) == (1, 5)

    with pytest.raises(ValueError, match="at least one"):
        worker.resolve_worker_queue("")
    with pytest.raises(ValueError, match="duplicate"):
        worker.resolve_worker_queue("0,4,0")
    with pytest.raises(ValueError, match="non-integer"):
        worker.resolve_worker_queue("0,nope")
    with pytest.raises(ValueError, match=r"\[0, 29\]"):
        worker.resolve_worker_queue("30")


def test_complete_production_config_is_shared_and_manifest_driven(tmp_path):
    corpus = _corpus(tmp_path)
    config = worker.build_production_config(tmp_path, corpus, graphs_per_batch=48)

    assert config.tasks == worker.TASKS
    assert config.train_seeds == (0, 1, 2)
    assert config.task_train_seeds == {}
    assert config.phases == ("scores", "carriage")
    assert config.resume is True
    assert config.force is False
    assert config.sizes.discovery_graphs == 48
    assert config.sizes.sources_per_graph == 6
    assert config.sizes.donors_per_source == 8
    assert config.sizes.bootstrap_replicates == 2_000
    assert config.bootstrap.replicates == 2_000
    assert config.execution.graphs_per_batch == 48
    assert config.execution.oom_backoff is True
    assert config.execution.replica_pair_budget is None
    assert len(config.checkpoints) == 30
    assert set(config.checkpoints) == {spec.key for spec in worker.WORKERS}
    assert config.checkpoints["zinc:0"].endswith("/checkpoints/zinc.dense.s0/best_available.ckpt")
    for task in worker.TASKS:
        expected_dataset = "zinc" if task.startswith("zinc") else "qm9"
        assert config.task_overrides[task]["dataset_dir"] == str(
            tmp_path / "datasets" / expected_dataset
        )


def test_auto_batch_profile_uses_all_graphs_on_a100_80gb():
    gib = 1024**3
    assert (
        worker.auto_graphs_per_batch(
            device_name="NVIDIA A100-SXM4-80GB", total_memory_bytes=80 * gib
        )
        == 48
    )
    assert (
        worker.auto_graphs_per_batch(
            device_name="NVIDIA A100-SXM4-40GB", total_memory_bytes=40 * gib
        )
        == 24
    )
    assert worker.auto_graphs_per_batch(device_name="NVIDIA L4", total_memory_bytes=22 * gib) == 12
    assert worker.auto_graphs_per_batch(device_name="Tesla T4", total_memory_bytes=16 * gib) == 8


def test_setup_generator_writes_30_commit_pinned_preindexed_workers(tmp_path):
    generated = worker.generate_worker_notebooks(
        NOTEBOOK,
        tmp_path / "workers",
        revision="a" * 40,
    )

    assert len(generated) == 30
    assert len({path.name for path in generated}) == 30
    for spec, path in zip(worker.WORKERS, generated):
        payload = json.loads(path.read_text(encoding="utf-8"))
        source = "".join("".join(cell.get("source", [])) for cell in payload["cells"])
        assert 'MODE = "worker"' in source
        assert f"WORKER_INDEX = {spec.index} " in source
        assert f'REPO_REVISION = "{"a" * 40}"' in source


def test_setup_generator_writes_four_commit_pinned_queue_notebooks(tmp_path):
    revision = "b" * 40
    generated = worker.generate_queue_notebooks(
        NOTEBOOK,
        tmp_path / "queues",
        revision=revision,
    )

    assert [path.name for path in generated] == [
        "queue_01_of_04.ipynb",
        "queue_02_of_04.ipynb",
        "queue_03_of_04.ipynb",
        "queue_04_of_04.ipynb",
    ]
    for indices, path in zip(worker.QUEUE_WORKER_INDICES, generated):
        payload = json.loads(path.read_text(encoding="utf-8"))
        source = "".join("".join(cell.get("source", [])) for cell in payload["cells"])
        queue = ",".join(str(index) for index in indices)
        assert 'MODE = "queue"' in source
        assert f"WORKER_INDEX = {indices[0]} " in source
        assert f'WORKER_INDICES = "{queue}"' in source
        assert f'REPO_REVISION = "{revision}"' in source


def test_setup_refresh_reuses_and_rehashes_existing_prepared_assets(tmp_path, monkeypatch):
    corpus = _corpus(tmp_path)
    calls = []

    monkeypatch.setattr(worker, "verify_archive_identity", lambda _root: tmp_path / "archive.tar")
    monkeypatch.setattr(
        worker,
        "read_archive_manifest",
        lambda _archive: dict(corpus.records),
    )
    monkeypatch.setattr(worker, "load_prepared_corpus", lambda _root: corpus)
    monkeypatch.setattr(
        worker,
        "_validate_extracted_checkpoints",
        lambda root, records, *, hash_all: calls.append((root, records, hash_all)),
    )
    datasets = {"roots": {"zinc": "zinc", "qm9": "qm9"}}
    monkeypatch.setattr(worker, "validate_shared_datasets", lambda _root: datasets)

    assert worker._reuse_prepared_setup_assets(tmp_path) == (corpus, datasets)
    assert calls == [(corpus.corpus_root, corpus.records, True)]


def test_checked_in_notebook_is_colab_ready_and_calls_frontend():
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "".join("".join(cell.get("source", [])) for cell in payload["cells"])

    assert payload["nbformat"] == 4
    assert payload["metadata"]["accelerator"] == "GPU"
    assert payload["metadata"]["colab"]["gpuType"] == "A100"
    assert 'MODE = "setup"' in source
    assert '"queue"' in source
    assert "WORKER_INDEX = 0" in source
    assert 'WORKER_INDICES = ""' in source
    assert "run_frontend(" in source
    assert "worker_indices=WORKER_INDICES" in source
    assert "GRAPHS_PER_BATCH or None" in source
    assert "RECLAIM_STALE_LOCK" in source
    assert "RECLAIM_WORKER_INDEX = -1" in source
    assert "reclaim_worker_index=RECLAIM_WORKER_INDEX" in source
    assert "importlib.invalidate_caches()" in source
    assert 'controller_module = "experiments.methodology.zinc_qm9_canonical_colab_worker"' in source
    assert "name == controller_module" in source
    assert 'vars(methodology_package).pop("zinc_qm9_canonical_colab_worker", None)' in source
    assert (
        'DRIVE_FOLDER = "/content/drive/MyDrive/graph_specialisation_metrics/multi_seed_models"'
    ) in source


def test_worker_drops_results_and_finalizer_uses_measurement_entrypoint():
    selected_source = inspect.getsource(worker._run_selected_worker_claimed)
    frontend_source = inspect.getsource(worker.run_frontend)

    assert "retain_results=False" in selected_source
    assert "release_component_memory()" in selected_source
    assert "finalize_measurement_run" in frontend_source
    assert "result = finalize_measurement_run(config)" in frontend_source


def test_worker_completion_uses_full_model_free_measurement_validator(monkeypatch):
    spec = worker.WORKERS[0]
    config = SimpleNamespace(
        fingerprint="f" * 64,
        strict_audits=False,
        phases=("scores", "carriage"),
    )
    checkpoint_sha = "c" * 64
    calls = []

    def fake_postflight(_config, task, seed):
        calls.append((task, seed))
        return {
            "task": task,
            "seed": seed,
            "checkpoint_sha256": checkpoint_sha,
            "graph_counts": {
                "expected": 48,
                "scores_semantic": 48,
                "scores_structural": 48,
                "carriage_semantic": 48,
                "carriage_structural": 48,
            },
            "artifacts": {
                stage: {
                    "path": f"/{stage}.pt",
                    "file_sha256": character * 64,
                    "contract_fingerprint": "d" * 64,
                    "event_manifest_hash": "e" * 64,
                }
                for stage, character in (("scores", "a"), ("carriage", "b"))
            },
            "audit_findings": [],
            "headline_eligible": True,
        }

    monkeypatch.setattr(
        "graph_specialisation_metrics.methodology.validate_measurement_worker",
        fake_postflight,
    )

    completion = worker.validate_worker_outputs(
        config,
        spec,
        expected_graphs=48,
        expected_checkpoint_sha256=checkpoint_sha,
    )

    assert calls == [(spec.task, spec.seed)]
    assert completion["schema"] == worker.COMPLETION_SCHEMA
    assert completion["checkpoint_sha256"] == checkpoint_sha
    assert set(completion["artifacts"]) == {"scores", "carriage"}
    assert completion["headline_eligible"] is True


def test_status_requires_current_completion_schema_and_all_artifacts(tmp_path):
    spec = worker.WORKERS[0]
    root = tmp_path / "canonical_outputs" / spec.task / f"seed_{spec.seed}"
    cache_paths = {
        "scores": root / "cache" / "scores" / "raw.pt",
        "carriage": root / "cache" / "carriage" / "fields.pt",
    }
    for path in (
        root / "protocol.json",
        root / "model.json",
        root / "audits.json",
        *cache_paths.values(),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    marker = {
        "schema": worker.COMPLETION_SCHEMA,
        "worker": {
            "index": spec.index,
            "task": spec.task,
            "seed": spec.seed,
            "run_id": spec.run_id,
        },
        "protocol_fingerprint": "f" * 64,
        "checkpoint_sha256": "c" * 64,
        "graphs": 48,
        "phases": ["scores", "carriage"],
        "artifacts": {
            stage: {
                "path": str(path),
                "file_sha256": character * 64,
                "contract_fingerprint": "d" * 64,
                "event_manifest_hash": "e" * 64,
            }
            for (stage, path), character in zip(cache_paths.items(), ("a", "b"))
        },
    }
    (root / "worker_complete.json").write_text(json.dumps(marker), encoding="utf-8")

    rows = worker.completion_status(tmp_path)
    assert rows[0]["complete"] is True
    assert sum(bool(row["complete"]) for row in rows) == 1

    cache_paths["carriage"].unlink()
    assert worker.completion_status(tmp_path)[0]["complete"] is False


def test_worker_queue_runs_sequentially_and_skips_only_validated_completion(tmp_path, monkeypatch):
    selected = worker.resolve_worker_queue("0,4,8")
    config = SimpleNamespace(sizes=SimpleNamespace(discovery_graphs=48))
    corpus = SimpleNamespace(drive_folder=tmp_path)
    calls = []

    @contextlib.contextmanager
    def fake_lock(_root, claim, *, reclaim=False):
        calls.append(("lock", claim, reclaim))
        yield tmp_path / claim

    def fake_verify(_corpus, spec):
        calls.append(("verify", spec.index))
        return f"sha-{spec.index}"

    def fake_completion(
        _config,
        spec,
        *,
        expected_graphs,
        expected_checkpoint_sha256,
    ):
        calls.append(("validate", spec.index, expected_graphs, expected_checkpoint_sha256))
        return {"worker_index": spec.index} if spec.index == 4 else None

    def fake_run(_config, spec, **kwargs):
        calls.append(
            (
                "run",
                spec.index,
                kwargs["checkpoint_digest"],
            )
        )
        return {"worker_index": spec.index}

    monkeypatch.setattr(worker, "drive_lock", fake_lock)
    monkeypatch.setattr(worker, "verify_selected_checkpoint", fake_verify)
    monkeypatch.setattr(worker, "validated_completion_record", fake_completion)
    monkeypatch.setattr(worker, "_run_selected_worker_claimed", fake_run)

    result = worker.run_worker_queue(
        config,
        corpus,
        selected,
        reclaim_worker_index=8,
    )

    assert result["worker_indices"] == [0, 4, 8]
    assert result["skipped_indices"] == [4]
    assert calls == [
        ("lock", "production_zinc_seed0", False),
        ("verify", 0),
        ("validate", 0, 48, "sha-0"),
        ("run", 0, "sha-0"),
        ("lock", "production_zinc_1hop_seed1", False),
        ("verify", 4),
        ("validate", 4, 48, "sha-4"),
        ("lock", "production_zinc_1hop_vnode_seed2", True),
        ("verify", 8),
        ("validate", 8, 48, "sha-8"),
        ("run", 8, "sha-8"),
    ]


def test_completion_skip_requires_matching_marker_and_fresh_cache_validation(tmp_path, monkeypatch):
    spec = worker.WORKERS[0]
    config = SimpleNamespace(
        output_dir=str(tmp_path),
        fingerprint="protocol-fingerprint",
        strict_audits=False,
        phases=("scores", "carriage"),
    )
    root = tmp_path / spec.task / f"seed_{spec.seed}"
    for relative in (
        "protocol.json",
        "model.json",
        "audits.json",
        "cache/scores/raw.pt",
        "cache/carriage/fields.pt",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    marker = {
        "schema": worker.COMPLETION_SCHEMA,
        "worker": {
            "index": spec.index,
            "task": spec.task,
            "seed": spec.seed,
            "run_id": spec.run_id,
        },
        "protocol_fingerprint": config.fingerprint,
        "checkpoint_sha256": "checkpoint-sha",
        "graphs": 48,
        "strict_audits": False,
        "phases": ["scores", "carriage"],
        "completed_at": "earlier",
    }
    (root / "worker_complete.json").write_text(json.dumps(marker), encoding="utf-8")
    validations = []
    monkeypatch.setattr(
        worker,
        "validate_worker_outputs",
        lambda *args, **kwargs: validations.append((args, kwargs)) or dict(marker),
    )

    assert (
        worker.validated_completion_record(
            config,
            spec,
            expected_graphs=48,
            expected_checkpoint_sha256="checkpoint-sha",
        )
        == marker
    )
    assert len(validations) == 1

    marker["graphs"] = 47
    (root / "worker_complete.json").write_text(json.dumps(marker), encoding="utf-8")
    assert (
        worker.validated_completion_record(
            config,
            spec,
            expected_graphs=48,
            expected_checkpoint_sha256="checkpoint-sha",
        )
        is None
    )
    assert len(validations) == 1


def test_worker_queue_fails_fast_without_starting_later_workers(tmp_path, monkeypatch):
    selected = worker.resolve_worker_queue("0,4,8")
    config = SimpleNamespace(sizes=SimpleNamespace(discovery_graphs=48))
    started = []

    @contextlib.contextmanager
    def fake_lock(_root, _claim, *, reclaim=False):
        del reclaim
        yield

    monkeypatch.setattr(worker, "drive_lock", fake_lock)
    monkeypatch.setattr(
        worker,
        "verify_selected_checkpoint",
        lambda _corpus, spec: f"sha-{spec.index}",
    )
    monkeypatch.setattr(
        worker,
        "validated_completion_record",
        lambda _config, _spec, **_kwargs: None,
    )

    def fake_run(_config, spec, **_kwargs):
        started.append(spec.index)
        if spec.index == 4:
            raise RuntimeError("synthetic worker failure")
        return {"worker_index": spec.index}

    monkeypatch.setattr(worker, "_run_selected_worker_claimed", fake_run)

    with pytest.raises(RuntimeError, match="synthetic worker failure"):
        worker.run_worker_queue(
            config,
            SimpleNamespace(drive_folder=tmp_path),
            selected,
        )
    assert started == [0, 4]
