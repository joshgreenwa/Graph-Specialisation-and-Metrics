"""Registration + checkpoint-discovery tests for the k-hop / VNode ZINC GRIT controls.

Mirrors tests/test_carriage_tasks.py: torch-free, exercises only the registry, the env hook
routing to the exact packaged training patch, and the recovery-
checkpoint discovery fallback that the k-hop runner's Colab-safe layout needs.
"""

from pathlib import Path

import pytest

from graph_specialisation_metrics.carriage import env
from graph_specialisation_metrics.carriage.tasks import get_task
from graph_specialisation_metrics.specialisation.model import GritHeadModel, _graph_major_order


@pytest.mark.parametrize(
    "name, params, drive, clone",
    [
        ("zinc_2hop", 473_473,
         "/content/drive/MyDrive/grit_zinc_2hop", "/content/GRIT_zinc_2hop"),
        ("zinc_1hop_vnode", 473_537,
         "/content/drive/MyDrive/grit_zinc_1hop_vnode", "/content/GRIT_zinc_1hop_vnode"),
        ("zinc_2hop_vnode", 473_537,
         "/content/drive/MyDrive/grit_zinc_2hop_vnode", "/content/GRIT_zinc_2hop_vnode"),
    ],
)
def test_khop_tasks_registered(name, params, drive, clone):
    t = get_task(name)
    # k-hop / VNode variants share the config the patch writes into their own clone.
    assert t.config_path == "configs/GRIT/zinc-GRIT-RRWP-khop.yaml"
    assert t.expected_params == params            # +64 for the VNode embedding
    assert t.drive_dir == drive
    assert t.grit_repo_dir == clone               # own clone: the patch is hops/vnode-specific
    assert len(t.env_hooks) == 1
    assert t.metric_higher_better is False
    assert t.metric_abort == 0.6
    # ZINC scalar regression uses the default whole-x-row content adapter.
    from graph_specialisation_metrics.carriage.content import FullNodeContentAdapter
    assert isinstance(t.content_adapter, FullNodeContentAdapter)


@pytest.mark.parametrize(
    "name, hops, vnode",
    [("zinc_2hop", 2, False), ("zinc_1hop_vnode", 1, True), ("zinc_2hop_vnode", 2, True)],
)
def test_khop_hook_routes_to_exact_training_patch(name, hops, vnode, monkeypatch, tmp_path):
    # The hook replays the packaged training patch with the task's hops + vnode.
    from graph_specialisation_metrics.grit_patches import khop_zinc

    calls = []
    monkeypatch.setattr(
        khop_zinc, "apply_khop_patch",
        lambda repo_dir, drive_dir, args: calls.append(
            (
                Path(repo_dir),
                Path(drive_dir),
                args.attention,
                int(args.hops),
                bool(args.global_vnode),
                args.expected_params,
                args.wandb_project,
            )
        ),
    )
    get_task(name).env_hooks[0](tmp_path)
    assert calls == [(tmp_path, tmp_path, "khop", hops, vnode, None, None)]


def test_checkpoint_discovery_prefers_best_recovery_checkpoint(tmp_path):
    # The k-hop runner writes stable recovery copies (best/latest.ckpt) instead of a GraphGym
    # ckpt/<epoch>.ckpt dir; best.ckpt (best validation) is preferred over latest.ckpt.
    rec = (tmp_path / "results" / "_recovery_checkpoints"
           / "seed0_ColabDrive.2hop.GRITwRRWP")
    rec.mkdir(parents=True)
    (rec / "latest.ckpt").touch()
    (rec / "best.ckpt").touch()

    chosen, epoch = env.find_checkpoint(tmp_path / "results")
    assert chosen == rec / "best.ckpt"
    assert epoch == -1


def test_checkpoint_discovery_falls_back_to_latest_when_no_best(tmp_path):
    rec = (tmp_path / "results" / "_recovery_checkpoints"
           / "seed0_ColabDrive.1hop.GRITwRRWP.VNode")
    rec.mkdir(parents=True)
    (rec / "latest.ckpt").touch()

    chosen, epoch = env.find_checkpoint(tmp_path / "results")
    assert chosen == rec / "latest.ckpt"
    assert epoch == -1


def test_standard_graphgym_ckpt_still_preferred_over_recovery(tmp_path):
    # A normal GraphGym ckpt/<epoch>.ckpt must still win when present (existing tasks).
    results = tmp_path / "results"
    std = results / "zinc-GRIT-RRWP-2hop" / "0" / "ckpt" / "1900.ckpt"
    std.parent.mkdir(parents=True)
    std.touch()
    rec = results / "_recovery_checkpoints" / "seed0_ColabDrive.2hop.GRITwRRWP"
    rec.mkdir(parents=True)
    (rec / "latest.ckpt").touch()

    chosen, epoch = env.find_checkpoint(results)
    assert chosen == std
    assert epoch == 1900


def test_grit_reregistration_replaces_previous_clone_registration():
    reg = pytest.importorskip("torch_geometric.graphgym.register")

    env.enable_grit_reregistration()
    old, new = object(), object()
    mapping = {"same_key": old}
    reg.register_base(mapping, "same_key", new)
    assert mapping["same_key"] is new


def test_vnode_rows_are_reordered_graph_major_for_replica_reshape():
    import torch

    # PyG real rows are graph-major, then GlobalVNode appends all virtual rows at the end.
    graph_index = torch.tensor([0, 0, 1, 1, 0, 1])
    order = _graph_major_order(graph_index)
    assert order.tolist() == [0, 1, 4, 2, 3, 5]
    assert graph_index[order].tolist() == [0, 0, 0, 1, 1, 1]


def test_capture_retains_and_groups_vnode_transport_before_pool_strip():
    import torch

    class Attention(torch.nn.Module):
        def forward(self, batch):
            return batch.x.reshape(-1, 1, 1), None

    attention = Attention()

    class Model(torch.nn.Module):
        def forward(self, batch):
            n = len(batch.x)
            batch.x = torch.cat([batch.x, torch.tensor([100., 200.])])
            batch.batch = torch.cat([batch.batch, torch.tensor([0, 1])])
            batch.real_node_mask = torch.arange(n + 2) < n
            attention(batch)  # capture the full internal transport site
            # Match GRIT: VNode entries are removed from batch.batch before capture() returns.
            batch.x = batch.x[batch.real_node_mask]
            batch.batch = batch.batch[batch.real_node_mask]
            return torch.zeros(2, 1), torch.zeros(2, 1)

    class Batch:
        x = torch.tensor([0., 1., 10., 11.])
        batch = torch.tensor([0, 0, 1, 1])

    gm = object.__new__(GritHeadModel)
    gm.model, gm.attn_layers = Model(), [attention]
    gm.L = gm.H = gm.dh = 1
    captured = gm.capture(Batch(), want_grad=False, include_virtual_transport=True)
    assert captured["wV"][0].reshape(-1).tolist() == [0., 1., 100., 10., 11., 200.]
    assert captured["node_graph"].tolist() == [0, 0, 0, 1, 1, 1]
