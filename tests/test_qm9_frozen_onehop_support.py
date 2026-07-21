import copy

import torch

from GRIT_QM9_gap import _upgrade_onehop_mask_to_explicit_support
from graph_specialisation_metrics.specialisation.scores import _perturb_mask_frozen


class _Data:
    def __init__(self):
        self.num_nodes = 3
        self.x = torch.arange(3, dtype=torch.float).view(-1, 1)
        # Bidirected path 0--1--2.  Swapping 0 and 1 is not an automorphism, so an
        # RRWP-derived support visibly moves while the frozen molecular mask does not.
        self.edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        self.edge_attr = torch.arange(4, dtype=torch.float).view(-1, 1)
        self.rrwp_index = torch.tensor(
            [
                [0, 1, 2, 0, 1, 1, 2],
                [0, 1, 2, 1, 0, 2, 1],
            ]
        )
        self.rrwp_val = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.5, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.5, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )

    def clone(self):
        return copy.deepcopy(self)


def _pairs(index):
    return {tuple(pair) for pair in index.t().tolist()}


def _rrwp_onehop_pairs(data):
    reachable = data.rrwp_val[:, :2].abs().sum(dim=-1) > 0
    return _pairs(data.rrwp_index[:, reachable])


def _edge_plus_self_pairs(data):
    return _pairs(data.edge_index) | {(node, node) for node in range(data.num_nodes)}


def test_qm9_frozen_intervention_requires_explicit_onehop_mask():
    base = _Data()
    pert = _perturb_mask_frozen(base, 0, 1)

    # On clean data, the old RRWP reachability route and edge_index+self are equivalent.
    assert _rrwp_onehop_pairs(base) == _edge_plus_self_pairs(base)
    # The intervention correctly freezes molecular wiring ...
    assert torch.equal(pert.edge_index, base.edge_index)
    assert _edge_plus_self_pairs(pert) == _edge_plus_self_pairs(base)
    # ... but deliberately transposes the RRWP payload.  Therefore an encoder that
    # re-derives support from rrwp_index/rrwp_val silently violates the frozen-mask contract.
    assert _rrwp_onehop_pairs(pert) != _rrwp_onehop_pairs(base)


def test_qm9_patch_migrates_existing_clone_to_edge_index_onehop_support(tmp_path):
    encoder = tmp_path / "rrwp_encoder.py"
    encoder.write_text(
        """\
class Encoder:
    def forward(self, batch):
        if self.max_hops is None:
            mask_index = batch.get(self.mask_index_name, None)
        else:
            needed_channels = self.max_hops + 1
""",
        encoding="utf-8",
    )

    assert _upgrade_onehop_mask_to_explicit_support(encoder) is True
    patched = encoder.read_text(encoding="utf-8")
    assert "if self.max_hops is None or self.max_hops == 1:" in patched
    assert "mask_index = batch.get(self.mask_index_name, None)" in patched
    assert "without silently rewiring attention" in patched
    compile(patched, str(encoder), "exec")

    # The migration must remain safe when analysis reuses an already-patched Colab clone.
    assert _upgrade_onehop_mask_to_explicit_support(encoder) is False

