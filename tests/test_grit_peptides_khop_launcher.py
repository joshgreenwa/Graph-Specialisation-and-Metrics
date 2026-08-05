from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


RUNNER_PATH = (
    Path(__file__).parents[1]
    / "experiments/peptides/training/GRIT_peptides_khop.py"
)
RUNNER_SPEC = importlib.util.spec_from_file_location("grit_peptides_khop", RUNNER_PATH)
assert RUNNER_SPEC is not None and RUNNER_SPEC.loader is not None
runner = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(runner)


@pytest.mark.parametrize("task", ["func", "struct"])
def test_requested_peptides_variants_have_isolated_drive_roots(task: str) -> None:
    cases = [
        (["--attention", "khop", "--hops", "1", "--rrwp-horizon", "1"], "1hop_localrrwp_h1"),
        (["--attention", "khop", "--hops", "1", "--global-vnode"], "1hop_vnode"),
        (["--attention", "khop", "--hops", "2"], "2hop"),
    ]
    for options, slug in cases:
        args = runner.parse_args(["--task", task, *options])
        assert runner.variant_slug(args) == slug
        assert args.drive_dir == Path(
            f"/content/drive/MyDrive/grit_peptides_{task}_{slug}"
        )
        assert args.dataset_dir == Path(
            f"/content/drive/MyDrive/grit_peptides_{task}_shared_data"
        )


def test_local_rrwp_preserves_task_specific_encoder_width() -> None:
    func = runner.parse_args(
        ["--task", "func", "--attention", "khop", "--hops", "1", "--rrwp-horizon", "1"]
    )
    struct = runner.parse_args(
        ["--task", "struct", "--attention", "khop", "--hops", "1", "--rrwp-horizon", "1"]
    )

    assert runner.expected_config_values(func)[("posenc_RRWP", "ksteps")] == 17
    assert runner.expected_config_values(struct)[("posenc_RRWP", "ksteps")] == 24
    assert runner.expected_config_values(func)[("posenc_RRWP", "local_horizon")] == 1
    assert runner.expected_config_values(struct)[("posenc_RRWP", "local_horizon")] == 1


@pytest.mark.parametrize(
    ("task", "hops"),
    [("func", 17), ("struct", 24)],
)
def test_hop_limit_tracks_available_rrwp_channels(task: str, hops: int) -> None:
    with pytest.raises(SystemExit):
        runner.parse_args(["--task", task, "--attention", "khop", "--hops", str(hops)])


def test_training_command_forwards_local_rrwp_horizon(tmp_path: Path) -> None:
    args = runner.parse_args(
        [
            "--task",
            "func",
            "--attention",
            "khop",
            "--hops",
            "1",
            "--rrwp-horizon",
            "1",
            "--drive-dir",
            str(tmp_path),
            "--dataset-dir",
            str(tmp_path / "shared_data"),
        ]
    )
    command = runner.build_train_command(args, Path("config.yaml"))

    assert command[command.index("posenc_RRWP.local_horizon") + 1] == "1"
