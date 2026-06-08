from dataclasses import dataclass

import torch

from graph_specialisation_metrics.core_interpretability_specialisation_metrics import (
    LayerFields,
    MetricOptions,
    OnlineAlphaAccumulator,
    SpecialisationMetricEngine,
    cosine_by_query,
    key_swap_field,
    masked_key_field,
    masked_softmax_permutation_weights,
    moved_mass_by_query,
    normalized_attention_entropy,
    output_from_fields,
    select_graph_indices,
)


def test_online_alpha_accumulator_matches_explicit_softmax_weights():
    torch.manual_seed(0)
    n_perms, batch, heads, nodes = 7, 2, 3, 5
    moved = torch.rand(n_perms, batch, heads, nodes)
    valid_alpha = torch.rand(n_perms, batch, heads, nodes) > 0.2
    valid_score = torch.rand(n_perms, batch, heads, nodes) > 0.1
    valid = valid_alpha & valid_score
    scores = {
        False: torch.rand(n_perms, batch, heads, nodes),
        True: torch.rand(n_perms, batch, heads, nodes) * 2.0 - 1.0,
    }
    follow = {
        False: torch.rand(n_perms, batch, heads, nodes),
        True: torch.rand(n_perms, batch, heads, nodes) * 2.0 - 1.0,
    }

    alpha_tau = 0.17
    alpha = masked_softmax_permutation_weights(moved, valid, alpha_tau)
    accumulator = OnlineAlphaAccumulator((False, True))
    for perm_idx in range(n_perms):
        accumulator.update(
            moved[perm_idx],
            valid_alpha[perm_idx],
            valid_score[perm_idx],
            {centered: scores[centered][perm_idx] for centered in (False, True)},
            {centered: follow[centered][perm_idx] for centered in (False, True)},
            alpha_tau,
        )

    final = accumulator.finalize()
    for centered in (False, True):
        expected_invariant = (alpha * scores[centered]).sum(dim=0)
        expected_follow = (alpha * follow[centered]).sum(dim=0)
        assert torch.allclose(final["invariant"][centered], expected_invariant, atol=1.0e-6)
        assert torch.allclose(final["follow"][centered], expected_follow, atol=1.0e-6)


def test_core_tensor_formulas_are_self_consistent():
    torch.manual_seed(1)
    batch, heads, nodes, dim = 1, 1, 4, 3
    attn = torch.tensor(
        [
            [
                [
                    [0.1, 0.6, 0.2, 0.1],
                    [0.3, 0.1, 0.5, 0.1],
                    [0.25, 0.25, 0.25, 0.25],
                    [0.7, 0.1, 0.1, 0.1],
                ]
            ]
        ]
    )
    msg = torch.randn(batch, heads, nodes, nodes, dim)
    mask = torch.ones(batch, heads, nodes, nodes, dtype=torch.bool)
    perm = torch.tensor([[1, 0, 2, 3]])

    ref = key_swap_field(attn, perm)
    assert torch.allclose(ref[..., 0], attn[..., 1])
    assert torch.allclose(ref[..., 1], attn[..., 0])

    score_follow = cosine_by_query(ref, ref, mask, centered=False)
    assert torch.allclose(score_follow, torch.ones_like(score_follow), atol=1.0e-6)

    block = torch.ones(batch, nodes, nodes, dtype=torch.bool)
    moved, valid = moved_mass_by_query(attn, mask, perm, block, field="routing")
    alpha = masked_softmax_permutation_weights(
        torch.stack([moved, moved]),
        torch.stack([valid, valid]),
        0.1,
    )
    assert torch.allclose(alpha.sum(dim=0)[valid], torch.ones_like(moved[valid]), atol=1.0e-6)

    var_attn = ref
    var_msg = key_swap_field(msg, perm)
    a0 = masked_key_field(attn, mask)
    a1 = masked_key_field(var_attn, mask)
    m0 = masked_key_field(msg, mask)
    m1 = masked_key_field(var_msg, mask)
    a_bar = 0.5 * (a0 + a1)
    m_bar = 0.5 * (m0 + m1)
    delta_a = ((a1 - a0).unsqueeze(-1) * m_bar).sum(dim=3)
    delta_m = (a_bar.unsqueeze(-1) * (m1 - m0)).sum(dim=3)
    delta = output_from_fields(var_attn, var_msg, mask) - output_from_fields(attn, msg, mask)
    assert torch.allclose(delta, delta_a + delta_m, atol=1.0e-5)

    uniform = torch.ones_like(attn) / nodes
    centered_uniform = cosine_by_query(uniform, uniform, mask, centered=True)
    assert torch.allclose(centered_uniform, torch.zeros_like(centered_uniform), atol=1.0e-6)

    entropy, entropy_valid = normalized_attention_entropy(uniform, mask)
    assert entropy_valid.all()
    assert torch.allclose(entropy, torch.ones_like(entropy), atol=1.0e-6)


@dataclass
class FakeBatch:
    node_type: torch.Tensor
    node_mask: torch.Tensor
    adj: torch.Tensor
    edge_value_mat: torch.Tensor
    degree: torch.Tensor
    spd: torch.Tensor
    rwse: torch.Tensor
    rrwp: torch.Tensor
    pair_xi: torch.Tensor
    edge_batch: torch.Tensor
    edge_src: torch.Tensor
    edge_dst: torch.Tensor
    edge_value: torch.Tensor
    num_graphs: int


class FakeCollector:
    def collect(self, batch: FakeBatch) -> list[LayerFields]:
        node = batch.node_type.float()
        graph_count, node_count = node.shape
        logits = (
            node[:, None, :, None] * 0.1
            + node[:, None, None, :] * 0.2
            + batch.degree[:, None, :, None] * 0.05
        )
        logits = logits.expand(-1, 2, -1, -1).clone()
        mask = batch.node_mask[:, None, :, None] & batch.node_mask[:, None, None, :]
        mask = mask.expand(-1, 2, -1, -1)
        attention = torch.softmax(logits.masked_fill(~mask, -1.0e9), dim=-1)
        message_base = torch.stack([node, batch.degree], dim=-1)
        message = message_base[:, None, None, :, :].expand(
            graph_count,
            2,
            node_count,
            -1,
            -1,
        )
        return [LayerFields(0, attention, message.clone(), mask, batch.node_mask)]


def make_fake_batch() -> FakeBatch:
    graph_count, node_count = 2, 5
    node_type = torch.tensor([[0, 1, 2, 3, 4], [2, 1, 0, 3, 4]])
    node_mask = torch.ones(graph_count, node_count, dtype=torch.bool)
    adj = torch.zeros(graph_count, node_count, node_count)
    for graph_idx in range(graph_count):
        adj[graph_idx, torch.arange(node_count - 1), torch.arange(1, node_count)] = 1
        adj[graph_idx, torch.arange(1, node_count), torch.arange(node_count - 1)] = 1

    edge_src = []
    edge_dst = []
    edge_batch = []
    edge_value = []
    for graph_idx in range(graph_count):
        src, dst = torch.nonzero(adj[graph_idx] > 0, as_tuple=True)
        edge_src.append(src)
        edge_dst.append(dst)
        edge_batch.append(torch.full_like(src, graph_idx))
        edge_value.append(torch.ones_like(src, dtype=torch.float32))

    return FakeBatch(
        node_type=node_type,
        node_mask=node_mask,
        adj=adj,
        edge_value_mat=adj.clone(),
        degree=adj.sum(dim=-1),
        spd=adj.long(),
        rwse=torch.randn(graph_count, node_count, 3),
        rrwp=torch.randn(graph_count, node_count, node_count, 3),
        pair_xi=torch.randn(graph_count, node_count, node_count, 2),
        edge_batch=torch.cat(edge_batch),
        edge_src=torch.cat(edge_src),
        edge_dst=torch.cat(edge_dst),
        edge_value=torch.cat(edge_value),
        num_graphs=graph_count,
    )


def test_metric_engine_smoke_covers_all_metric_families_and_preserves_graph_indices():
    options = MetricOptions(
        metrics=("routing", "transport", "output"),
        interventions=("content", "structure"),
        blocks=("all", "local", "global"),
        centered=(False, True),
        num_permutations=6,
        alpha_tau=0.1,
        seed=0,
    )
    engine = SpecialisationMetricEngine(FakeCollector(), options)
    engine.compute_batch(make_fake_batch(), graph_indices=[10, 20])
    result = engine.results()

    assert result.per_graph_rows
    assert result.summary_rows
    assert {row["graph_index"] for row in result.per_graph_rows}.issubset({10, 20})
    summary_metrics = {row["metric"] for row in result.summary_rows}
    assert "routing_follow" in summary_metrics
    assert "transport_invariant" in summary_metrics
    assert "output_routing_responsibility" in summary_metrics
    assert "global_routing_gate" in summary_metrics
    assert "attention_entropy" in summary_metrics


def test_select_graph_indices_is_sorted_reproducible_and_handles_full_dataset():
    dataset = list(range(10))
    selected = select_graph_indices(dataset, num_graphs=4, seed=123)
    assert selected == sorted(selected)
    assert selected == select_graph_indices(dataset, num_graphs=4, seed=123)
    assert select_graph_indices(dataset, num_graphs=0, seed=123) == list(range(10))
    assert select_graph_indices(dataset, num_graphs=10, seed=123) == list(range(10))
