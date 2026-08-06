"""Canonical registration and patch-contract tests for all QM9 gap controls."""

from pathlib import Path

import pytest

from graph_specialisation_metrics.carriage import qm9_env
from graph_specialisation_metrics.carriage.tasks import get_task
from graph_specialisation_metrics.methodology.tasks import TASKS as CANONICAL_TASKS

QM9_VARIANTS = (
    (
        "qm9_gap_dense",
        "dense",
        1,
        -1,
        False,
        472_769,
        "/content/drive/MyDrive/grit_qm9_gap_dense",
        "/content/GRIT_qm9_gap_dense",
    ),
    (
        "qm9_gap_1hop",
        "khop",
        1,
        -1,
        False,
        472_769,
        "/content/drive/MyDrive/grit_qm9_gap_1hop_real",
        "/content/GRIT_qm9_gap_1hop",
    ),
    (
        "qm9_gap_1hop_local",
        "khop",
        1,
        1,
        False,
        472_769,
        "/content/drive/MyDrive/grit_qm9_gap_1hop_localrrwp",
        "/content/GRIT_qm9_gap_1hop_local",
    ),
    (
        "qm9_gap_2hop",
        "khop",
        2,
        -1,
        False,
        472_769,
        "/content/drive/MyDrive/grit_qm9_gap_2hop",
        "/content/GRIT_qm9_gap_2hop",
    ),
    (
        "qm9_gap_1hop_vnode",
        "khop",
        1,
        -1,
        True,
        472_833,
        "/content/drive/MyDrive/grit_qm9_gap_1hop_vnode",
        "/content/GRIT_qm9_gap_1hop_vnode",
    ),
    (
        "qm9_gap_2hop_vnode",
        "khop",
        2,
        -1,
        True,
        472_833,
        "/content/drive/MyDrive/grit_qm9_gap_2hop_vnode",
        "/content/GRIT_qm9_gap_2hop_vnode",
    ),
)


@pytest.mark.parametrize(
    "name,attention,hops,rrwp_horizon,vnode,params,drive,clone",
    QM9_VARIANTS,
)
def test_all_qm9_controls_are_registered_for_canonical_analysis(
    name,
    attention,
    hops,
    rrwp_horizon,
    vnode,
    params,
    drive,
    clone,
):
    del attention, hops, rrwp_horizon
    task = get_task(name)

    assert name in CANONICAL_TASKS
    assert task.config_path == "configs/GRIT/qm9-gap-GRIT-RRWP.yaml"
    assert task.expected_params == params
    assert task.drive_dir == drive
    assert task.dataset_dir == "/content/drive/MyDrive/grit_qm9_gap_data"
    assert task.grit_repo_dir == clone
    assert task.metric_higher_better is False
    assert task.metric_abort == 0.5
    assert len(task.env_hooks) == 1
    assert CANONICAL_TASKS[name].virtual_node is vnode
    assert "rrwp_attention_edge_index" in CANONICAL_TASKS[name].fixed_support_fields
    assert CANONICAL_TASKS[name].carrier_policy == (
        "real_nodes_plus_internal_vnode" if vnode else "real_nodes"
    )


@pytest.mark.parametrize(
    "name,attention,hops,rrwp_horizon,vnode,params,drive,clone",
    QM9_VARIANTS,
)
def test_all_qm9_hooks_replay_the_complete_training_patch_contract(
    name,
    attention,
    hops,
    rrwp_horizon,
    vnode,
    params,
    drive,
    clone,
    monkeypatch,
    tmp_path,
):
    del params, drive, clone
    calls = []

    def apply_patch(repo_dir, drive_dir, args):
        calls.append(
            (
                "apply",
                Path(repo_dir),
                Path(drive_dir),
                args.attention,
                int(args.hops),
                int(args.rrwp_horizon),
                bool(args.global_vnode),
                int(args.batch_size),
                int(args.epochs),
                int(args.warmup_epochs),
                args.expected_params,
                args.wandb_project,
            )
        )

    def verify_patch(repo_dir):
        calls.append(("verify", Path(repo_dir)))

    monkeypatch.setattr(
        qm9_env,
        "_import_qm9_patch",
        lambda: (apply_patch, verify_patch),
    )

    get_task(name).env_hooks[0](tmp_path)

    assert calls == [
        (
            "apply",
            tmp_path,
            tmp_path,
            attention,
            hops,
            rrwp_horizon,
            vnode,
            128,
            300,
            10,
            None,
            None,
        ),
        ("verify", tmp_path),
    ]


@pytest.mark.parametrize("rrwp_horizon", [-2, 21])
def test_qm9_analysis_hook_rejects_an_invalid_rrwp_horizon(rrwp_horizon):
    with pytest.raises(ValueError, match="rrwp_horizon"):
        qm9_env.make_qm9_hook("khop", rrwp_horizon=rrwp_horizon)
