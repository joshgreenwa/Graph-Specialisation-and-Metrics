from pathlib import Path

import pytest

from graph_specialisation_metrics.carriage import env, onehop_env
from graph_specialisation_metrics.carriage.tasks import get_task


@pytest.mark.parametrize(
    "name,expected_params",
    [
        ("zinc_1hop", 473_473),
        ("zinc_2hop", 473_473),
        ("zinc_1hop_vnode", 473_537),
        ("zinc_2hop_vnode", 473_537),
        ("qm9_gap_1hop", 472_769),
        ("qm9_gap_1hop_vnode", 472_833),
    ],
)
def test_requested_receptive_field_tasks_have_registered_drive_checkpoints(
    name,
    expected_params,
):
    task = get_task(name)
    assert task.drive_dir
    assert task.expected_params == expected_params
    assert task.grit_repo_dir
    assert len(task.env_hooks) == 1


def test_zinc_1hop_local_task_matches_onehop_except_local_rrwp_reconstruction():
    standard = get_task("zinc_1hop")
    local = get_task("zinc_1hop_local")

    assert local.config_path == "configs/GRIT/zinc-GRIT-RRWP-1hop-localrrwp.yaml"
    assert local.drive_dir == "/content/drive/MyDrive/grit_zinc_1hop_localrrwp"
    assert local.expected_params == standard.expected_params == 473_473
    assert local.metric_fn.__func__ is standard.metric_fn.__func__
    assert local.metric_higher_better == standard.metric_higher_better
    assert local.metric_abort == standard.metric_abort
    assert local.grit_repo_dir != standard.grit_repo_dir
    assert len(local.env_hooks) == 1


def test_zinc_1hop_local_hook_routes_to_exact_training_patch(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        onehop_env,
        "apply_onehop_localrrwp_patch",
        lambda repo_dir: calls.append(Path(repo_dir)),
    )

    get_task("zinc_1hop_local").env_hooks[0](tmp_path)

    assert calls == [tmp_path]


def test_checkpoint_discovery_accepts_localrrwp_training_layout(tmp_path):
    results = tmp_path / "results"
    ckpt = (
        results
        / "zinc-GRIT-RRWP-1hop-localrrwp-ColabDrive.1hopLocalRRWP.GRITwRRWP"
        / "0"
        / "ckpt"
        / "1900.ckpt"
    )
    ckpt.parent.mkdir(parents=True)
    ckpt.touch()

    chosen, epoch = env.find_checkpoint(results)

    assert chosen == ckpt
    assert epoch == 1900
