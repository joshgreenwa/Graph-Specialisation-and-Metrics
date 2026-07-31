from pathlib import Path

from graph_specialisation_metrics.carriage import env, onehop_env
from graph_specialisation_metrics.carriage.tasks import get_task


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


def test_zinc_1hop_localrrwp_alias_preserves_checkpoint_contract():
    historical = get_task("zinc_1hop_local")
    descriptive = get_task("zinc_1hop_localrrwp")

    assert descriptive.drive_dir == historical.drive_dir
    assert descriptive.config_path == historical.config_path
    assert descriptive.expected_params == historical.expected_params
    assert descriptive.grit_repo_dir == historical.grit_repo_dir


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
