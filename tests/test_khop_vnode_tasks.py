"""Registration + checkpoint-discovery tests for the k-hop / VNode ZINC GRIT controls.

Mirrors tests/test_carriage_tasks.py: torch-free, exercises only the registry, the env hook
routing to the exact packaged training patch, and the recovery-
checkpoint discovery fallback that the k-hop runner's Colab-safe layout needs.
"""

import importlib.util
from pathlib import Path

import pytest

from graph_specialisation_metrics.carriage import env
from graph_specialisation_metrics.carriage.tasks import get_task
from graph_specialisation_metrics.methodology.interventions import (
    StructuralAuditError,
    structural_donor_swap,
)
from graph_specialisation_metrics.methodology.tasks import (
    TASKS as CANONICAL_TASKS,
)
from graph_specialisation_metrics.methodology.tasks import (
    ZINC_FROZEN_KHOP_ADAPTER_VERSION,
)
from graph_specialisation_metrics.specialisation.model import GritHeadModel, _graph_major_order

RUNNER_PATH = (
    Path(__file__).parents[1]
    / "src/graph_specialisation_metrics/grit_patches/khop_zinc.py"
)
RUNNER_SPEC = importlib.util.spec_from_file_location("grit_khop_zinc", RUNNER_PATH)
assert RUNNER_SPEC is not None and RUNNER_SPEC.loader is not None
khop_zinc = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(khop_zinc)


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
    canonical = CANONICAL_TASKS[name]
    assert "rrwp_attention_edge_index" in canonical.fixed_support_fields
    assert canonical.adapter_version == ZINC_FROZEN_KHOP_ADAPTER_VERSION


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


def test_old_khop_encoder_is_upgraded_to_frozen_support(tmp_path):
    encoder = tmp_path / "rrwp_encoder.py"
    encoder.write_text(
        """\
        if self.max_hops is None:
            mask_index = batch.get(self.mask_index_name, None)
        else:
            needed_channels = self.max_hops + 1  # identity + walks of length 1..k
            if self.max_hops < 1 or needed_channels > raw_rrwp_val.size(1):
                raise ValueError(
                    f"max_hops={self.max_hops} requires {needed_channels} RRWP channels; "
                    f"found {raw_rrwp_val.size(1)}."
                )
            reachable = raw_rrwp_val[:, :needed_channels].abs().sum(dim=-1) > 0
            mask_index = rrwp_idx[:, reachable]
""",
        encoding="utf-8",
    )

    assert khop_zinc._upgrade_onehop_mask_to_explicit_support(encoder)
    assert khop_zinc._upgrade_khop_mask_to_frozen_support(encoder)
    upgraded = encoder.read_text(encoding="utf-8")
    assert "self.max_hops is None or self.max_hops == 1" in upgraded
    assert 'batch.get("rrwp_attention_edge_index", None)' in upgraded
    assert "Backward-compatible clean-forward fallback" in upgraded


def test_khop_patch_verifier_requires_onehop_frozen_support_marker(tmp_path):
    verifier_source = RUNNER_PATH.read_text(encoding="utf-8")
    assert '"self.max_hops is None or self.max_hops == 1"' in verifier_source


@pytest.mark.parametrize("hops", [1, 2])
def test_frozen_support_is_clean_forward_equivalent_to_training_mask(hops):
    torch = pytest.importorskip("torch")
    adjacency = torch.tensor(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.5, 0.0, 0.5, 0.0],
            [0.0, 0.5, 0.0, 0.5],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    powers = [torch.eye(4), adjacency]
    for _ in range(2):
        powers.append(powers[-1] @ adjacency)
    pe = torch.stack(powers, dim=-1)

    # Training built rrwp_index/rrwp_val from all non-zero RRWP rows, then selected rows
    # reachable within k. The refined transform selects those same clean coordinates before
    # an intervention can alter their payload.
    all_row, all_col = pe.abs().sum(dim=-1).gt(0).nonzero(as_tuple=True)
    rrwp_index = torch.stack([all_col, all_row])
    rrwp_val = pe[all_row, all_col]
    legacy = rrwp_index[:, rrwp_val[:, : hops + 1].abs().sum(dim=-1).gt(0)]

    frozen_mask = pe[..., : hops + 1].abs().sum(dim=-1).gt(0)
    frozen_row, frozen_col = frozen_mask.nonzero(as_tuple=True)
    frozen = torch.stack([frozen_col, frozen_row])

    assert torch.equal(frozen, legacy)


@pytest.mark.parametrize("name", ["zinc_2hop", "zinc_1hop_vnode", "zinc_2hop_vnode"])
def test_structural_rrwp_swap_preserves_frozen_zinc_attention_support(name):
    torch = pytest.importorskip("torch")
    Data = pytest.importorskip("torch_geometric.data").Data
    task = CANONICAL_TASKS[name]
    rrwp_index = torch.tensor(
        [[0, 0, 1, 1, 2, 2], [0, 1, 0, 2, 1, 2]], dtype=torch.long
    )
    base = Data(
        x=torch.tensor([[1], [2], [3]], dtype=torch.long),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long),
        edge_attr=torch.ones(4, 1, dtype=torch.long),
        rrwp_index=rrwp_index,
        rrwp_val=torch.tensor([[1.0], [0.2], [0.3], [0.4], [0.5], [1.0]]),
        rrwp=torch.arange(6, dtype=torch.float32).reshape(3, 2),
        rrwp_attention_edge_index=torch.tensor(
            [[0, 0, 1, 1, 1, 2, 2], [0, 1, 0, 1, 2, 1, 2]], dtype=torch.long
        ),
        y=torch.tensor([0.0]),
    )

    event = structural_donor_swap(
        base, 0, 2, task=task, duplicate_tolerance=1.0e-7
    )

    assert not torch.equal(event.rrwp_val, base.rrwp_val)
    assert torch.equal(event.edge_index, base.edge_index)
    assert torch.equal(
        event.rrwp_attention_edge_index, base.rrwp_attention_edge_index
    )


@pytest.mark.parametrize("name", ["zinc_2hop", "zinc_1hop_vnode", "zinc_2hop_vnode"])
def test_zinc_khop_structural_swap_fails_without_frozen_attention_support(name):
    torch = pytest.importorskip("torch")
    Data = pytest.importorskip("torch_geometric.data").Data
    base = Data(
        x=torch.tensor([[1], [2]], dtype=torch.long),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        edge_attr=torch.ones(2, 1, dtype=torch.long),
        rrwp_index=torch.tensor([[0, 1], [0, 1]], dtype=torch.long),
        rrwp_val=torch.ones(2, 1),
        rrwp=torch.ones(2, 1),
        y=torch.tensor([0.0]),
    )

    with pytest.raises(StructuralAuditError, match="frozen rrwp_attention_edge_index"):
        structural_donor_swap(
            base,
            0,
            1,
            task=CANONICAL_TASKS[name],
            duplicate_tolerance=1.0e-7,
        )


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
