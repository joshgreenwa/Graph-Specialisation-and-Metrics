from __future__ import annotations

import csv
import importlib.util
import json
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "experiments/grit_hpc/bin/grit_hpc.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("grit_hpc_test_module", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def hpc():
    return load_runner()


@pytest.mark.parametrize(
    ("variant", "expected"),
    [
        ("dense", ("dense", 1, False, -1)),
        ("dense_vnode", ("dense", 1, True, -1)),
        ("2hop", ("khop", 2, False, -1)),
        ("3hop_vnode", ("khop", 3, True, -1)),
        ("1hop_localrrwp_h1", ("khop", 1, False, 1)),
        ("2hop_vnode_localrrwp_h2", ("khop", 2, True, 2)),
    ],
)
def test_parse_variant(hpc, variant, expected):
    assert hpc.parse_variant(variant) == expected


def test_default_grid_has_one_unique_row_per_model(hpc):
    jobs = hpc.jobs_from_grid(
        hpc.TASKS,
        ["dense", "1hop", "1hop_vnode", "2hop", "2hop_vnode"],
        [0, 1],
    )
    assert len(jobs) == 40
    assert len({job.run_id for job in jobs}) == 40
    assert jobs[0].run_id == "zinc.dense.s0"
    assert jobs[-1].run_id == "peptides_struct.2hop_vnode.s1"


def test_validation_rejects_unsupported_hops_and_zinc_local_rrwp(hpc):
    with pytest.raises(ValueError, match="1..16"):
        hpc.jobs_from_grid(["peptides_func"], ["17hop"], [0])
    with pytest.raises(ValueError, match="ZINC"):
        hpc.jobs_from_grid(["zinc"], ["1hop_localrrwp_h1"], [0])


@pytest.mark.parametrize(
    ("task", "required"),
    [
        ("zinc", []),
        ("qm9_gap", ["--rrwp-horizon", "2"]),
        ("peptides_func", ["--task", "func", "--rrwp-horizon", "2"]),
        ("peptides_struct", ["--task", "struct", "--rrwp-horizon", "2"]),
    ],
)
def test_runner_argv_isolated_and_install_free(hpc, tmp_path, task, required):
    horizon = -1 if task == "zinc" else 2
    job = hpc.Job(f"{task}.2hop_vnode.s7", task, "khop", 2, True, horizon, 7)
    argv = hpc.runner_argv(
        job,
        output_dir=tmp_path / "out" / job.run_id,
        dataset_dir=tmp_path / "data" / task,
        repo_dir=tmp_path / "scratch" / job.run_id,
        grit_source="/shared/pristine_GRIT",
        num_threads=8,
        accelerator="cuda:0",
        auto_resume=True,
        wandb=True,
        wandb_project="grit-multi-dataset",
    )
    assert "--skip-install" in argv
    assert "--skip-editable-install" in argv
    assert argv[argv.index("--recovery-ckpt-period") + 1] == "10"
    assert argv[argv.index("--wandb-project") + 1] == "grit-multi-dataset"
    assert "--global-vnode" in argv
    assert argv[argv.index("--repo-url") + 1] == "/shared/pristine_GRIT"
    assert argv[argv.index("--attention") + 1] == "khop"
    assert argv[argv.index("--hops") + 1] == "2"
    for token in required:
        assert token in argv


def test_manifest_round_trip_and_dry_run_reserves_output(hpc, tmp_path):
    job = hpc.Job("qm9_gap.1hop_vnode_localrrwp_h1.s0", "qm9_gap", "khop", 1, True, 1, 0)
    manifest = tmp_path / "jobs.jsonl"
    hpc.write_manifest(manifest, [job])
    assert hpc.read_manifest(manifest) == [job]

    marker = hpc.dataset_ready_path(tmp_path / "datasets", job.task)
    hpc.atomic_write_json(marker, {"task": job.task})
    hpc.main(
        [
            "run",
            "--manifest", str(manifest),
            "--array-index", "0",
            "--dataset-root", str(tmp_path / "datasets"),
            "--output-root", str(tmp_path / "outputs"),
            "--scratch-root", str(tmp_path / "scratch"),
            "--dry-run",
        ]
    )
    spec_path = tmp_path / "outputs" / job.run_id / "hpc_job.json"
    assert json.loads(spec_path.read_text(encoding="utf-8")) == hpc.asdict(job)
    attempts = list((spec_path.parent / "hpc_attempts").glob("*.json"))
    assert len(attempts) == 1
    assert json.loads(attempts[0].read_text(encoding="utf-8"))["run_id"] == job.run_id


def test_tracking_ledger_maps_every_slurm_handle(hpc, tmp_path):
    jobs = hpc.jobs_from_grid(["zinc"], ["dense", "1hop_vnode"], [0, 1])
    manifest = tmp_path / "jobs.jsonl"
    ledger = tmp_path / "jobs.tsv"
    hpc.write_manifest(manifest, jobs)
    hpc.main(
        [
            "write-tracking",
            "--manifest", str(manifest),
            "--slurm-array-job-id", "98765",
            "--output-root", str(tmp_path / "outputs"),
            "--log-root", str(tmp_path / "logs"),
            "--output", str(ledger),
        ]
    )
    rows = list(csv.DictReader(ledger.open(encoding="utf-8"), delimiter="\t"))
    assert [row["slurm_handle"] for row in rows] == ["98765_0", "98765_1", "98765_2", "98765_3"]
    assert [row["run_id"] for row in rows] == [job.run_id for job in jobs]


def test_status_tracking_joins_scheduler_state_to_run_id(hpc, tmp_path, monkeypatch, capsys):
    jobs = hpc.jobs_from_grid(["zinc"], ["dense"], [0, 1])
    manifest = tmp_path / "jobs.jsonl"
    ledger = tmp_path / "jobs.tsv"
    hpc.write_manifest(manifest, jobs)
    hpc.main(
        [
            "write-tracking",
            "--manifest", str(manifest),
            "--slurm-array-job-id", "98765",
            "--output-root", str(tmp_path / "outputs"),
            "--log-root", str(tmp_path / "logs"),
            "--output", str(ledger),
        ]
    )
    capsys.readouterr()

    def fake_run(command, **_kwargs):
        if command[0] == "sacct":
            return SimpleNamespace(stdout="98765_0|COMPLETED|05:30:00|06:00:00|0:0\n")
        return SimpleNamespace(stdout="98765|1|RUNNING|01:12|06:00:00|gpu-node-7\n")

    monkeypatch.setattr(hpc.subprocess, "run", fake_run)
    hpc.main(["status-tracking", "--tracking", str(ledger)])
    output = capsys.readouterr().out
    assert "zinc.dense.s0" in output and "COMPLETED" in output
    assert "zinc.dense.s1" in output and "RUNNING" in output
    assert "Summary: COMPLETED=1, RUNNING=1" in output


def test_empty_manifest_is_rejected(hpc, tmp_path):
    with pytest.raises(ValueError, match="empty manifest"):
        hpc.write_manifest(tmp_path / "empty.jsonl", [])


def test_csd3_launcher_stays_within_400_gpu_hours():
    script = (ROOT / "experiments/grit_hpc/bin/submit_csd3_60.sh").read_text(encoding="utf-8")
    assert "--seeds 0,1,2" in script
    assert "--variants dense,1hop,1hop_vnode,2hop,2hop_vnode" in script
    assert "--array=0-59%60" in script
    assert 'TRAIN_TIME_LIMIT="${GRIT_JOB_TIME_LIMIT:-06:00:00}"' in script
    assert 'STAGE_TIME_LIMIT="${GRIT_STAGE_TIME_LIMIT:-01:00:00}"' in script
    assert "REQUESTED_SECONDS=$((60 * TRAIN_SECONDS + 4 * STAGE_SECONDS))" in script
    assert "REQUESTED_SECONDS > BUDGET_SECONDS" in script
    assert "mlmi-jgg45-sl2-gpu" in script
    assert "/rds/user/jgg45/hpc-work" in script


def test_gpu_worker_requires_staged_dataset(hpc, tmp_path):
    job = hpc.Job("zinc.dense.s0", "zinc", "dense", 1, False, -1, 0)
    manifest = tmp_path / "jobs.jsonl"
    hpc.write_manifest(manifest, [job])
    with pytest.raises(RuntimeError, match="not staged"):
        hpc.main(
            [
                "run",
                "--manifest", str(manifest),
                "--array-index", "0",
                "--dataset-root", str(tmp_path / "datasets"),
                "--output-root", str(tmp_path / "outputs"),
                "--scratch-root", str(tmp_path / "scratch"),
                "--dry-run",
            ]
        )
