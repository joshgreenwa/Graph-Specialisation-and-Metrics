"""Contract tests for the two ZINC checkpoint-trajectory Colab notebooks."""

from __future__ import annotations

import io
import json
import sys
import tarfile
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.methodology import zinc_checkpoint_trajectory_colab as trajectory
from graph_specialisation_metrics.methodology.cache import (
    CacheContract,
    CanonicalCache,
    checkpoint_sha256,
)
from graph_specialisation_metrics.methodology.tasks import get_task

NOTEBOOKS = {
    "dense": ROOT / "experiments" / "methodology" / "zinc_dense_checkpoint_trajectory_colab.ipynb",
    "1hop": ROOT / "experiments" / "methodology" / "zinc_1hop_checkpoint_trajectory_colab.ipynb",
}


def _record(architecture: str, epoch: int, checkpoint: Path) -> trajectory.TrajectoryCheckpoint:
    return trajectory.TrajectoryCheckpoint(
        architecture=architecture,
        task=trajectory.ARCHITECTURE_TASK[architecture],
        seed=0,
        epoch=epoch,
        relative_path=f"{architecture}/epoch{epoch}.ckpt",
        sha256=checkpoint_sha256(checkpoint),
        bytes=checkpoint.stat().st_size,
    )


def _prepared(tmp_path: Path) -> trajectory.PreparedTrajectory:
    corpus = tmp_path / "corpus"
    records = {}
    for architecture in trajectory.ARCHITECTURE_TASK:
        for epoch in trajectory.EPOCHS:
            checkpoint = corpus / architecture / f"epoch{epoch}.ckpt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(f"{architecture}:{epoch}".encode())
            records[(architecture, epoch)] = _record(architecture, epoch, checkpoint)
    dataset = tmp_path / "datasets" / "zinc"
    dataset.mkdir(parents=True)
    return trajectory.PreparedTrajectory(tmp_path, corpus, records, dataset)


def _tar_member(bundle: tarfile.TarFile, name: str, payload: bytes) -> None:
    member = tarfile.TarInfo(name)
    member.size = len(payload)
    bundle.addfile(member, io.BytesIO(payload))


def test_registered_archive_and_manifest_cover_exact_two_by_six_grid(tmp_path):
    assert trajectory.ARCHIVE_NAME == "zinc_dense_1hop_seed0_trajectory.tar"
    assert trajectory.ARCHIVE_BYTES == 71_086_080
    assert (
        trajectory.ARCHIVE_SHA256
        == "e8b37688bea8abefba11fe972345c1bf539cdb98d55b87fd625fe2f0e4d33634"
    )
    assert trajectory.EPOCHS == (10, 100, 250, 500, 1_000, 1_990)
    assert set(trajectory.EXPECTED_CHECKPOINT_SHA256) == {
        (architecture, epoch) for architecture in ("dense", "1hop") for epoch in trajectory.EPOCHS
    }

    manifest = ["model\tseed\tepoch\tcheckpoint\tsha256\n"]
    for (architecture, epoch), digest in trajectory.EXPECTED_CHECKPOINT_SHA256.items():
        manifest.append(f"{architecture}\t0\t{epoch}\t{architecture}/epoch{epoch}.ckpt\t{digest}\n")
    archive = tmp_path / "trajectory.tar"
    with tarfile.open(archive, "w") as bundle:
        _tar_member(
            bundle,
            f"{trajectory.ARCHIVE_ROOT}/manifest.tsv",
            "".join(manifest).encode(),
        )
        for architecture, epoch in trajectory.EXPECTED_CHECKPOINT_SHA256:
            _tar_member(
                bundle,
                f"{trajectory.ARCHIVE_ROOT}/{architecture}/epoch{epoch}.ckpt",
                f"checkpoint:{architecture}:{epoch}".encode(),
            )
    records = trajectory.read_archive_manifest(archive)
    assert len(records) == 12
    assert records[("dense", 1_990)].task == "zinc"
    assert records[("1hop", 10)].task == "zinc_1hop"


def test_epoch_configs_are_scores_only_isolated_and_stable_within_architecture(tmp_path):
    prepared = _prepared(tmp_path)
    dense_configs = [
        trajectory.build_epoch_config(
            prepared, prepared.records[("dense", epoch)], graphs_per_batch=48
        )
        for epoch in trajectory.EPOCHS
    ]
    assert {config.phases for config in dense_configs} == {("scores",)}
    assert {config.fingerprint for config in dense_configs} == {dense_configs[0].fingerprint}
    assert len({config.output_dir for config in dense_configs}) == 6
    assert all(
        "score_trajectory_outputs/dense/epoch_" in config.output_dir for config in dense_configs
    )
    assert all(config.execution.graphs_per_batch == 48 for config in dense_configs)
    assert all(config.sizes.discovery_graphs == 48 for config in dense_configs)
    assert all(config.sizes.bootstrap_replicates == 2_000 for config in dense_configs)

    one_hop = trajectory.build_epoch_config(
        prepared, prepared.records[("1hop", 10)], graphs_per_batch=48
    )
    assert one_hop.fingerprint != dense_configs[0].fingerprint
    assert one_hop.tasks == ("zinc_1hop",)
    assert list(one_hop.checkpoints) == ["zinc_1hop:0"]


def _write_valid_score_worker(config, record: trajectory.TrajectoryCheckpoint) -> None:
    from graph_specialisation_metrics.methodology.cache import atomic_json
    from graph_specialisation_metrics.methodology.protocol import PROTOCOL_VERSION, stable_hash

    root = config.root / record.task / "seed_0"
    root.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(config.checkpoints[f"{record.task}:0"])
    geometry = {
        "layers": 2,
        "heads": 3,
        "head_width": 4,
        "hidden_width": 12,
        "outputs": 1,
    }
    splits = {
        "discovery": list(range(48)),
        "causal": [100],
        "clean_ablation": [101],
        "semantic_donor_pool": [200, 201],
        "same_index_space": False,
        "seed": config.analysis_seed,
    }
    protocol = config.record()
    protocol.update(
        {
            "repository_commit": "test",
            "execution_mode": "isolated-seed-worker",
            "worker_task": record.task,
            "worker_seed": 0,
        }
    )
    atomic_json(root / "protocol.json", protocol)
    atomic_json(
        root / "model.json",
        {
            "protocol_version": PROTOCOL_VERSION,
            "task": record.task,
            "train_seed": 0,
            "checkpoint": str(checkpoint),
            "checkpoint_epoch": -1,
            "checkpoint_sha256": checkpoint_sha256(checkpoint),
            "task_adapter_version": get_task(record.task).adapter_version,
            "output_representation": get_task(record.task).output.representation,
            "sigma": [1.0],
            "model_geometry": geometry,
            "splits": splits,
            "validation_metric": 0.2,
            "test_metric": 0.3,
            "parameter_count": 123,
            "canonical_audits": {"failures": []},
        },
    )
    atomic_json(
        root / "audits.json",
        {
            "protocol_version": PROTOCOL_VERSION,
            "task": record.task,
            "train_seed": 0,
            "strict_audits": False,
            "phases": ["scores"],
            "findings": [],
            "headline_eligible": True,
        },
    )
    contract = CacheContract(
        protocol_fingerprint=config.fingerprint,
        task=record.task,
        task_adapter_version=get_task(record.task).adapter_version,
        checkpoint_sha256=checkpoint_sha256(checkpoint),
        train_seed=0,
        model_geometry=geometry,
        output_representation=get_task(record.task).output.representation,
        sigma=(1.0,),
        split_fingerprint=stable_hash(splits),
        event_manifest_hash="a" * 24,
        donors_per_source=config.sizes.donors_per_source,
        source_cap=config.sizes.sources_per_graph,
        bootstrap_seed=config.bootstrap.rng_seed,
        bootstrap_replicates=config.bootstrap.replicates,
    )
    raw = np.arange(1, 7, dtype=float).reshape(2, 3)
    graphs = {graph_id: raw + graph_id / 1_000 for graph_id in range(48)}
    CanonicalCache(config.root, contract).save(
        "scores",
        "raw",
        {
            "protocol_version": PROTOCOL_VERSION,
            "manifest_hash": "a" * 24,
            "channels": {
                "semantic": {"raw": raw, "graph_scores": graphs},
                "structural": {"raw": raw * 2, "graph_scores": graphs},
            },
        },
    )


def test_scores_only_postflight_checks_full_cache_contract(tmp_path):
    prepared = _prepared(tmp_path)
    record = prepared.records[("dense", 10)]
    config = trajectory.build_epoch_config(prepared, record, graphs_per_batch=4, accelerator="cpu")
    _write_valid_score_worker(config, record)

    validated = trajectory.validate_score_output(config, record)
    assert validated["epoch"] == 10
    assert validated["raw"]["semantic"].shape == (2, 3)
    assert validated["checkpoint_sha256"] == record.sha256
    assert validated["audit_findings"] == 0

    model_path = config.root / "zinc" / "seed_0" / "model.json"
    model = json.loads(model_path.read_text())
    model["task_adapter_version"] = "stale-adapter"
    model_path.write_text(json.dumps(model))
    with pytest.raises(RuntimeError, match="stale task adapter"):
        trajectory.validate_score_output(config, record)


def _fake_validated(record: trajectory.TrajectoryCheckpoint) -> dict:
    raw = np.arange(1, 7, dtype=float).reshape(2, 3) * (1 + record.epoch / 1_000)
    return {
        "architecture": record.architecture,
        "task": record.task,
        "seed": record.seed,
        "epoch": record.epoch,
        "checkpoint_sha256": record.sha256,
        "protocol_fingerprint": "b" * 24,
        "split_fingerprint": "c" * 24,
        "event_manifest_hash": f"{record.epoch:024d}",
        "artifact_path": f"/cache/{record.architecture}/{record.epoch}/raw.pt",
        "artifact_sha256": "d" * 64,
        "contract_fingerprint": "e" * 24,
        "raw": {"semantic": raw.copy(), "structural": raw.copy() * 2},
        "model_geometry": {
            "layers": 2,
            "heads": 3,
            "head_width": 4,
            "hidden_width": 12,
            "outputs": 1,
        },
        "validation_metric": 0.2,
        "test_metric": 0.3,
        "audit_findings": 0,
        "headline_eligible": True,
    }


def test_sequential_runner_freshly_validates_and_skips_complete_epochs(tmp_path, monkeypatch):
    prepared = _prepared(tmp_path)
    monkeypatch.setattr(
        trajectory,
        "validate_score_output",
        lambda config, record, checkpoint_digest=None: _fake_validated(record),
    )
    import graph_specialisation_metrics.methodology.runner as runner_module

    monkeypatch.setattr(
        runner_module,
        "run_worker",
        lambda *args, **kwargs: pytest.fail("validated epochs must not rerun"),
    )
    result = trajectory.run_architecture(prepared, "dense", graphs_per_batch=4, accelerator="cpu")
    assert result["skipped_epochs"] == list(trajectory.EPOCHS)
    for epoch in trajectory.EPOCHS:
        marker = (
            trajectory.epoch_output_dir(tmp_path, "dense", epoch)
            / "zinc"
            / "seed_0"
            / "trajectory_score_complete.json"
        )
        assert json.loads(marker.read_text())["schema"] == trajectory.COMPLETION_SCHEMA


def test_plotter_writes_two_channel_distributions_and_machine_readable_tables(
    tmp_path, monkeypatch
):
    prepared = _prepared(tmp_path)
    monkeypatch.setattr(
        trajectory,
        "validate_score_output",
        lambda config, record, checkpoint_digest=None: _fake_validated(record),
    )
    result = trajectory.plot_architecture(prepared, "1hop", graphs_per_batch=4, accelerator="cpu")
    files = result["files"]
    for name in ("png", "pdf", "long_csv", "summary_csv"):
        assert Path(files[name]).is_file()
        assert Path(files[name]).stat().st_size > 0
    long_rows = Path(files["long_csv"]).read_text().splitlines()
    summary_rows = Path(files["summary_csv"]).read_text().splitlines()
    assert len(long_rows) == 1 + 6 * 2 * 2 * 3
    assert len(summary_rows) == 1 + 6 * 2
    assert result["architecture"] == "1hop"
    assert result["epochs"] == list(trajectory.EPOCHS)


@pytest.mark.parametrize(("architecture", "notebook"), NOTEBOOKS.items())
def test_notebooks_are_two_direct_run_all_entry_points(architecture, notebook):
    payload = json.loads(notebook.read_text())
    source = "\n".join(
        "".join(cell.get("source", ())) for cell in payload["cells"] if cell["cell_type"] == "code"
    )
    assert 'MODE = "run"' in source
    assert f'ARCHITECTURE = "{architecture}"' in source
    assert "multi_seed_models/multiple_checkpoints_zinc" in source
    assert "zinc_checkpoint_trajectory_colab import run_frontend" in source
    assert (
        'controller_module = "experiments.methodology.zinc_checkpoint_trajectory_colab"' in source
    )
    assert "graphs_per_batch=(GRAPHS_PER_BATCH or None)" in source
    assert payload["metadata"]["accelerator"] == "GPU"
