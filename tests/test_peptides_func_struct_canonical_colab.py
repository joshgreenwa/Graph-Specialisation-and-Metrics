"""Contracts for the two-lane Peptides canonical Colab controller."""

from __future__ import annotations

import dataclasses
import json
from contextlib import contextmanager
from pathlib import Path

import pytest

from experiments.methodology import peptides_func_struct_canonical_colab as controller
from graph_specialisation_metrics.carriage.tasks import get_task as get_grit_task
from graph_specialisation_metrics.methodology.tasks import get_task

NOTEBOOKS = {
    "func": Path(__file__).parents[1]
    / "experiments/methodology/peptides_func_canonical_colab.ipynb",
    "struct": Path(__file__).parents[1]
    / "experiments/methodology/peptides_struct_canonical_colab.ipynb",
}
PINNED_REVISION = "296b39b044c7b08f358ff44f178865b43422ad61"
PINNED_BRANCH = "expansion/carriage_experiments"


def _records():
    rows = []
    for worker in controller.WORKERS:
        variant = worker.run_id.split(".")[1]
        task = "peptides_func" if worker.dataset == "func" else "peptides_struct"
        rows.append(
            {
                "run_id": worker.run_id,
                "task": task,
                "variant": variant,
                "seed": worker.seed,
                "archive_checkpoint": f"checkpoints/{worker.run_id}/best_available.ckpt",
                "source_checkpoint_sha256": f"{worker.index + 1:064x}",
                "source_checkpoint_bytes": 1_000 + worker.index,
                "exact_global_best": True,
                "selected_epoch": 100 + worker.index,
                "selected_validation": 0.2 + worker.index / 1_000,
                "selected_test": 0.3 + worker.index / 1_000,
                "selection_metric": "ap" if worker.dataset == "func" else "mae",
                "selection_direction": "max" if worker.dataset == "func" else "min",
            }
        )
    return rows


def _corpus(tmp_path: Path) -> controller.PreparedCorpus:
    records = controller.validate_archive_manifest(_records())
    return controller.PreparedCorpus(tmp_path, tmp_path / controller.ARCHIVE_ROOT, records)


def test_archive_identity_and_exact_two_lane_worker_grid():
    assert controller.ARCHIVE_NAME == "peptides_func_struct_best_checkpoints.tar"
    assert controller.ARCHIVE_BYTES == 162_631_680
    assert (
        controller.ARCHIVE_SHA256
        == "d6e5923249326f0d2498d7f5ca996ec2a25c4d213c41cbfdf9c7a28224ad3456"
    )
    assert controller.DEFAULT_DRIVE_FOLDER == Path(
        "/content/drive/MyDrive/graph_specialisation_metrics/multi_seed_models/"
        "peptides_func_struct_checkpoints"
    )
    assert len(controller.TASKS) == 10
    assert len(controller.WORKERS) == 30
    assert [worker.index for worker in controller.WORKERS] == list(range(30))
    assert tuple(worker.index for worker in controller.DATASET_WORKERS["func"]) == tuple(range(15))
    assert tuple(worker.index for worker in controller.DATASET_WORKERS["struct"]) == tuple(
        range(15, 30)
    )


def test_manifest_is_explicitly_mapped_and_rejects_non_best_checkpoint():
    records = controller.validate_archive_manifest(_records())
    assert records["peptides_func.dense.s0"]["source_checkpoint_sha256"] == f"{1:064x}"
    assert records["peptides_struct.2hop_vnode.s2"]["selected_epoch"] == 129

    malformed = _records()
    malformed[0] = {**malformed[0], "exact_global_best": False}
    with pytest.raises(RuntimeError, match="invalid checkpoint selection"):
        controller.validate_archive_manifest(malformed)


def test_all_unified_peptides_tasks_replay_expected_geometry_and_support():
    expected = {
        "peptides_func_dense": (443_434, False, False),
        "peptides_func_1hop": (443_434, False, False),
        "peptides_func_1hop_vnode": (443_530, True, False),
        "peptides_func_2hop": (443_434, False, True),
        "peptides_func_2hop_vnode": (443_530, True, True),
        "peptides_struct_dense": (449_579, False, False),
        "peptides_struct_1hop": (449_579, False, False),
        "peptides_struct_1hop_vnode": (449_675, True, False),
        "peptides_struct_2hop": (449_579, False, True),
        "peptides_struct_2hop_vnode": (449_675, True, True),
    }
    for name, (parameters, vnode, frozen_support) in expected.items():
        grit = get_grit_task(name)
        task = get_task(name)
        assert grit.expected_params == parameters
        assert task.virtual_node is vnode
        assert ("rrwp_attention_edge_index" in task.fixed_support_fields) is frozen_support
        assert task.adapter_version.startswith("canonical-grit-peptides-")
        assert grit.config_path.endswith("-GRIT-RRWP-custom.yaml")


def test_complete_config_is_shared_by_both_lanes_and_is_a100_optimised(tmp_path: Path):
    corpus = _corpus(tmp_path)
    config = controller.build_production_config(tmp_path, corpus, graphs_per_batch=16)

    assert config.tasks == controller.TASKS
    assert config.train_seeds == (0, 1, 2)
    assert config.phases == ("scores", "carriage")
    assert len(config.checkpoints) == 30
    assert config.sizes.discovery_graphs == 48
    assert config.sizes.sources_per_graph == 6
    assert config.sizes.donors_per_source == 8
    assert config.bootstrap.replicates == 2_000
    assert config.execution.graphs_per_batch == 16
    assert config.execution.oom_backoff is True
    assert config.execution.jacobian_output_chunk == 11
    assert config.num_threads == 8
    for task in controller.TASKS:
        override = config.task_overrides[task]
        dataset = "func" if task.startswith("peptides_func_") else "struct"
        assert override["dataset_dir"] == str(tmp_path / "datasets" / dataset)
        assert override["analysis_split_limits"] == {
            "train": 2_000,
            "val": 136,
            "test": 136,
            "seed": 31_415,
        }
        assert override["disable_metric_abort_guard"] is True


def test_a100_80gb_starts_at_sixteen_graph_groups_with_oom_backoff():
    gib = 1024**3
    assert (
        controller.auto_graphs_per_batch(
            device_name="NVIDIA A100-SXM4-80GB", total_memory_bytes=80 * gib
        )
        == 16
    )
    assert (
        controller.auto_graphs_per_batch(
            device_name="NVIDIA A100-SXM4-40GB", total_memory_bytes=40 * gib
        )
        == 8
    )


def test_subset_metric_verification_records_full_split_provenance(tmp_path: Path):
    corpus = _corpus(tmp_path)
    config = controller.build_production_config(tmp_path, corpus, graphs_per_batch=2)
    worker = controller.WORKERS[0]
    model_path = controller._completion_path(config, worker).parent / "model.json"
    model_path.parent.mkdir(parents=True)
    record = corpus.records[worker.run_id]
    model_path.write_text(
        json.dumps(
            {
                "test_metric": record["selected_test"],
                "validation_metric": record["selected_validation"],
            }
        ),
        encoding="utf-8",
    )
    verified = controller._metric_verification_record(config, corpus, worker)
    assert verified["scope"] == "analysis_subset"
    assert verified["subset_test"] == record["selected_test"]
    assert verified["archive_full_split_test"] == record["selected_test"]

    model_path.write_text(
        json.dumps(
            {
                "test_metric": float("nan"),
                "validation_metric": record["selected_validation"],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="non-finite"):
        controller._metric_verification_record(config, corpus, worker)


def test_frontend_rejects_cross_lane_stale_lock_reclaim(tmp_path: Path, monkeypatch):
    corpus = _corpus(tmp_path)
    config = controller.build_production_config(tmp_path, corpus, graphs_per_batch=2)
    with pytest.raises(ValueError, match="not in the func lane"):
        controller.run_dataset_queue(
            config,
            corpus,
            "func",
            reclaim_worker_index=20,
        )


def test_setup_reclaim_is_explicit_and_repairs_under_the_setup_claim(tmp_path: Path, monkeypatch):
    calls = []
    prepared = object()

    @contextmanager
    def fake_lock(root, name, *, reclaim=False):
        calls.append((root, name, reclaim))
        yield root

    monkeypatch.setattr(controller, "drive_lock", fake_lock)
    monkeypatch.setattr(
        controller,
        "load_prepared_corpus",
        lambda _root: (_ for _ in ()).throw(RuntimeError("not ready")),
    )
    monkeypatch.setattr(controller, "prepare_corpus", lambda _root: prepared)

    assert controller.ensure_prepared_corpus(tmp_path, reclaim_setup_lock=True) is prepared
    assert calls == [(tmp_path, "peptides_corpus_setup", True)]


def test_completion_status_uses_dataset_specific_cache_paths(tmp_path: Path):
    rows = controller.completion_status(tmp_path)
    assert len(rows) == 30
    assert rows[0] == {
        "index": 0,
        "dataset": "func",
        "task": "peptides_func_dense",
        "seed": 0,
        "scores": False,
        "carriage": False,
        "complete": False,
    }
    assert rows[-1]["task"] == "peptides_struct_2hop_vnode"


def test_worker_record_is_json_serializable():
    assert json.loads(json.dumps(dataclasses.asdict(controller.WORKERS[0])))["index"] == 0


@pytest.mark.parametrize("dataset", ["func", "struct"])
def test_checked_in_notebooks_are_a100_ready_pinned_dataset_lanes(dataset: str):
    payload = json.loads(NOTEBOOKS[dataset].read_text(encoding="utf-8"))
    code_cells = [
        "".join(cell.get("source", []))
        for cell in payload["cells"]
        if cell.get("cell_type") == "code"
    ]
    source = "".join(code_cells)

    assert payload["nbformat"] == 4
    assert payload["metadata"]["accelerator"] == "GPU"
    assert payload["metadata"]["colab"]["gpuType"] == "A100"
    assert 'MODE = "run"' in source
    assert f'DATASET = "{dataset}"' in source
    assert f'REPO_BRANCH = "{PINNED_BRANCH}"' in source
    assert f'REPO_REVISION = "{PINNED_REVISION}"' in source
    assert 'GITHUB_SECRET = "dissertation_key"' in source
    assert "from google.colab import userdata" in source
    assert "token = userdata.get(GITHUB_SECRET)" in source
    assert "Colab secret {GITHUB_SECRET!r} is missing or empty" in source
    assert "peptides_func_struct_checkpoints" in source
    assert "GRAPHS_PER_BATCH = 0" in source
    assert "RECLAIM_SETUP_LOCK = False" in source
    assert "RECLAIM_WORKER_INDEX = -1" in source
    assert "public_url = 'https:' + '//github.com/" in source
    assert "public_url = 'https://github.com/" not in source
    assert "GIT_CONFIG_VALUE_0" in source
    assert "Authorization: Basic {credential}" in source
    assert "'--branch', REPO_BRANCH, '--single-branch', '--no-tags'" in source
    assert "remote_action = 'set-url' if 'origin' in remotes else 'add'" in source
    assert "'remote', remote_action, 'origin', public_url" in source
    assert "'config', '--local', '--unset-all', 'http.https://github.com/.extraheader'" in source
    assert "f'+refs/heads/{REPO_BRANCH}:{remote_ref}'" in source
    assert "'merge-base', '--is-ancestor', REPO_REVISION, remote_ref" in source
    assert "'checkout', '--force', '-B', REPO_BRANCH, REPO_REVISION" in source
    assert "remote_url != public_url" in source
    assert "repo.exists() and not (repo / '.git').is_dir()" in source
    assert "shutil.rmtree(repo)" in source
    assert "name == 'graph_specialisation_metrics'" in source
    assert "controller_path.is_relative_to(repo.resolve())" in source
    assert "run_frontend(" in source
    assert "graphs_per_batch=GRAPHS_PER_BATCH or None" in source
    assert "reclaim_setup_lock=RECLAIM_SETUP_LOCK" in source
    assert "vars(methodology_package).pop('peptides_func_struct_canonical_colab'" in source
    for index, cell in enumerate(code_cells):
        compile(cell, f"{dataset}-cell-{index}", "exec")
