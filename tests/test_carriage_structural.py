import copy

import torch

from graph_specialisation_metrics.carriage import structural


class _Data:
    def __init__(self):
        self.x = torch.tensor([[10.0], [20.0], [30.0]])
        self.edge_index = torch.tensor([[0, 1, 1], [1, 0, 2]])
        self.edge_attr = torch.tensor([[1.0], [2.0], [3.0]])
        self.rrwp_index = torch.tensor([[0, 1, 2], [0, 2, 1]])
        self.rrwp_val = torch.tensor([[0.1], [0.2], [0.3]])
        self.rrwp_local_edge_index = self.edge_index.clone()

    def clone(self):
        return copy.deepcopy(self)


def test_local_rrwp_support_is_relabelled_by_structural_transposition():
    base = _Data()

    pert = structural.perturb(base, 0, 2, "transposition")

    expected = structural._relabel_uv(base.rrwp_local_edge_index, 0, 2)
    assert torch.equal(pert.rrwp_local_edge_index, expected)
    assert torch.equal(pert.x, base.x)
    structural.verify_perturbation(base, pert, 0, 2, "transposition")


def test_full_relabel_moves_local_rrwp_support_and_content_together():
    base = _Data()

    relabelled = structural.full_relabel(base, 0, 2)

    assert torch.equal(
        relabelled.rrwp_local_edge_index,
        structural._relabel_uv(base.rrwp_local_edge_index, 0, 2),
    )
    assert torch.equal(relabelled.x, structural._swap_rows(base.x, 0, 2))


def test_single_node_copy_handles_index_only_local_rrwp_support():
    base = _Data()

    pert = structural.perturb(base, 0, 1, "single_node")
    expected, _ = structural._copy_incidence(base.rrwp_local_edge_index, 0, 1)

    assert torch.equal(pert.rrwp_local_edge_index, expected)
    assert torch.equal(pert.x, base.x)
