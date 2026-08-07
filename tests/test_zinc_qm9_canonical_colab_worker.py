"""Contract tests for the emergency 30-checkpoint Colab frontend."""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

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


def test_checked_in_notebook_is_colab_ready_and_calls_frontend():
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "".join("".join(cell.get("source", [])) for cell in payload["cells"])

    assert payload["nbformat"] == 4
    assert payload["metadata"]["accelerator"] == "GPU"
    assert payload["metadata"]["colab"]["gpuType"] == "A100"
    assert 'MODE = "setup"' in source
    assert "WORKER_INDEX = 0" in source
    assert "run_frontend(" in source
    assert "GRAPHS_PER_BATCH or None" in source
    assert "RECLAIM_STALE_LOCK" in source
    assert (
        'DRIVE_FOLDER = "/content/drive/MyDrive/graph_specialisation_metrics/'
        'multi_seed_models"'
    ) in source


def test_worker_drops_results_and_finalizer_uses_measurement_entrypoint():
    selected_source = inspect.getsource(worker.run_selected_worker)
    frontend_source = inspect.getsource(worker.run_frontend)

    assert "retain_results=False" in selected_source
    assert "release_component_memory()" in selected_source
    assert "finalize_measurement_run" in frontend_source
    assert "result = finalize_measurement_run(config)" in frontend_source
