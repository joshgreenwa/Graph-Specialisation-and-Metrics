import math

import numpy as np
import torch

from graph_specialisation_metrics.method_adapters import (
    DirectLinkAdapter,
    StepByStepChainAdapter,
    chain_graph_with_branches,
)
from graph_specialisation_metrics.method_core import (
    GraphBatchView,
    above_null_margin,
    effective_rank,
    edge_index_from_edges,
    integrated_gradients_output,
    minimum_vertex_cut,
    non_additivity_ratio,
    swap_source_influence,
)


def test_integrated_gradients_reconstructs_linear_function():
    weight = torch.tensor([[1.0, -2.0], [0.5, 3.0]])
    x = torch.tensor([[2.0, -1.0], [4.0, 0.5]])
    baseline = torch.zeros_like(x)

    def predict(x_new):
        return (x_new * weight).sum().view(1)

    ig = integrated_gradients_output(predict, x, baseline, steps=8)
    assert torch.allclose(ig, x * weight, atol=1.0e-6)
    assert torch.allclose(ig.sum(), predict(x) - predict(baseline), atol=1.0e-6)


def test_swap_source_influence_preserves_topology_and_changes_content():
    class SumAdapter:
        def predict(self, graph):
            return graph.x.sum().view(1)

    edge_index = edge_index_from_edges(3, [(0, 1), (1, 2)])
    graph = GraphBatchView(x=torch.arange(6, dtype=torch.float32).view(3, 2), edge_index=edge_index)
    before = graph.edge_index.clone()
    _ = swap_source_influence(SumAdapter(), graph, source=0, partner=2, normalize=False)
    assert torch.equal(graph.edge_index, before)
    assert torch.equal(graph.x[0], torch.tensor([0.0, 1.0]))


def test_minimum_vertex_cut_handles_chain_cycle_and_disconnected():
    chain = GraphBatchView(torch.ones(5, 1), edge_index_from_edges(5, [(0, 1), (1, 2), (2, 3), (3, 4)]))
    assert minimum_vertex_cut(chain, 0, 4) in ([1], [2], [3])

    cycle = GraphBatchView(torch.ones(4, 1), edge_index_from_edges(4, [(0, 1), (1, 2), (2, 3), (3, 0)]))
    assert len(minimum_vertex_cut(cycle, 0, 2)) == 2

    disconnected = GraphBatchView(torch.ones(4, 1), edge_index_from_edges(4, [(0, 1), (2, 3)]))
    assert minimum_vertex_cut(disconnected, 0, 3) == []


def test_analytic_mediator_patching_direct_vs_composed():
    graph = chain_graph_with_branches(length=7, branch_attachments=[2, 3, 4], seed=1)
    readout = torch.ones(graph.x.size(1))
    direct = DirectLinkAdapter(0, 6, readout)
    step = StepByStepChainAdapter(0, 6, readout, list(range(7)))

    def retained(adapter, clamp):
        base = graph.clone_with(x=graph.x.clone())
        base.x[0] = torch.zeros_like(base.x[0])
        unclamped = float(adapter.predict(graph) - adapter.predict(base))
        clamped = float(adapter.patch_hidden_states(graph, clamp).prediction - adapter.patch_hidden_states(base, clamp).prediction)
        return abs(clamped) / max(abs(unclamped), 1.0e-12)

    assert math.isclose(retained(direct, [3]), 1.0, rel_tol=1.0e-6)
    assert retained(step, [3]) == 0.0
    branch = graph.metadata["branch_nodes"][0]
    assert math.isclose(retained(step, [branch]), 1.0, rel_tol=1.0e-6)


def test_effective_rank_and_above_null_margin_on_planted_matrix():
    rng = np.random.default_rng(3)
    a = rng.normal(size=(20, 2))
    b = rng.normal(size=(2, 20))
    matrix = a @ b
    assert effective_rank(matrix, energy=0.99) == 2
    assert above_null_margin(matrix, permutations=8, seed=4) > 0.0


def test_non_additivity_separates_additive_and_interacting():
    additive = non_additivity_ratio(delta_a=2.0, delta_b=-0.5, delta_ab=1.5)
    interacting = non_additivity_ratio(delta_a=2.0, delta_b=-0.5, delta_ab=4.0)
    assert additive == 0.0
    assert interacting > 0.5
