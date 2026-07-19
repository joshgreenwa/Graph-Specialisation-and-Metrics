import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from graph_specialisation_metrics.specialisation.effective_transport import (
    apply_effective_treatment,
    effective_transport_decomposition,
    graphwise_broadcast_residual,
    grit_attention_components,
    paired_bootstrap_summary,
    run_effective_transport,
    support_marginal_static_attention,
)


def _scatter(values, index, size):
    out = values.new_zeros((size,) + tuple(values.shape[1:]))
    out.index_add_(0, index, values)
    return out


def _normalised_attention(logits, dst, n):
    result = torch.zeros_like(logits)
    for node in range(n):
        mask = dst == node
        result[mask] = torch.softmax(logits[mask], dim=0)
    return result


def test_grit_attention_components_reconstructs_content_and_verow_exactly():
    torch.manual_seed(4)
    n, heads, head_dim = 3, 2, 2
    edge_index = torch.tensor(
        [[0, 1, 2, 0, 2, 1], [0, 0, 1, 1, 2, 2]], dtype=torch.long
    )
    src, dst = edge_index
    attention = _normalised_attention(torch.randn(src.numel(), heads), dst, n)
    v_h = torch.randn(n, heads, head_dim)
    edge_state = torch.randn(src.numel(), heads, head_dim)
    ve_row = torch.randn(head_dim, heads, head_dim)

    value_edge = torch.einsum("ehd,dhc->ehc", edge_state, ve_row)
    message_content = attention.unsqueeze(-1) * v_h[src]
    message_edge = attention.unsqueeze(-1) * value_edge
    wv_content = _scatter(message_content, dst, n)
    wv_edge = _scatter(message_edge, dst, n)
    wv = wv_content + wv_edge

    module = SimpleNamespace(edge_enhance=True, VeRow=ve_row)
    batch = SimpleNamespace(
        edge_index=edge_index,
        attn=attention.unsqueeze(-1),
        V_h=v_h,
        get=lambda key, default=None: getattr(batch, key, default),
    )
    captured = grit_attention_components(module, batch, (wv, edge_state.flatten(1)))

    assert torch.allclose(captured["value_edge"], value_edge)
    assert torch.allclose(captured["wv_content"], wv_content)
    assert torch.allclose(captured["wv_edge"], wv_edge)
    assert torch.allclose(captured["wv_reconstructed"], wv, atol=1e-7)
    assert torch.allclose(captured["pair_message"], message_content + message_edge)


def test_grit_attention_components_handles_disabled_edge_enhancement():
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    attention = torch.ones(2, 1, 1)
    v_h = torch.tensor([[[2.0]], [[3.0]]])
    wv = torch.tensor([[[3.0]], [[2.0]]])
    module = SimpleNamespace(edge_enhance=False)
    batch = SimpleNamespace(
        edge_index=edge_index,
        attn=attention,
        V_h=v_h,
        get=lambda key, default=None: getattr(batch, key, default),
    )

    captured = grit_attention_components(module, batch, (wv, None))

    assert torch.count_nonzero(captured["value_edge"]) == 0
    assert torch.equal(captured["wv_reconstructed"], wv)


def test_dense_static_null_is_repeated_column_mean_and_preserves_marginals():
    torch.manual_seed(12)
    n, heads = 4, 3
    src = torch.arange(n).repeat(n)
    dst = torch.arange(n).repeat_interleave(n)
    edge_index = torch.stack([src, dst])
    matrix = torch.softmax(torch.randn(n, n, heads), dim=1)
    attention = matrix.reshape(n * n, heads)

    static, diagnostics = support_marginal_static_attention(
        attention, edge_index, n, return_diagnostics=True
    )
    expected_matrix = matrix.mean(dim=0, keepdim=True).expand(n, -1, -1)

    assert torch.allclose(static.reshape(n, n, heads), expected_matrix, atol=2e-6)
    assert torch.allclose(_scatter(static, dst, n), torch.ones(n, heads), atol=2e-6)
    assert torch.allclose(
        _scatter(static, src, n), _scatter(attention, src, n), atol=2e-6
    )
    assert diagnostics["max_destination_mass_error"] < 2e-6
    assert diagnostics["max_source_mass_error"] < 2e-6


def test_sparse_static_null_keeps_edge_slots_and_both_marginals():
    torch.manual_seed(17)
    n, heads = 5, 2
    # Directed cycle plus self-loops: sparse but connected and strictly feasible.
    src = torch.cat([torch.arange(n), torch.arange(n)])
    dst = torch.cat([torch.arange(n), torch.roll(torch.arange(n), shifts=-1)])
    edge_index = torch.stack([src, dst])
    attention = _normalised_attention(torch.randn(src.numel(), heads), dst, n)

    static = support_marginal_static_attention(attention, edge_index, n)

    assert static.shape == attention.shape
    assert torch.all(static >= 0)
    assert torch.allclose(_scatter(static, dst, n), _scatter(attention, dst, n), atol=2e-6)
    assert torch.allclose(_scatter(static, src, n), _scatter(attention, src, n), atol=2e-6)
    # The null is genuinely support constrained: it has only the original 2n slots.
    assert static.size(0) == 2 * n


def test_effective_decomposition_reconstructs_projection_and_energy_cross_term():
    torch.manual_seed(23)
    node_batch = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)
    n, heads, head_dim, output_dim = 5, 2, 3, 4
    u_content = torch.randn(n, heads, head_dim)
    u_edge = 0.4 * torch.randn(n, heads, head_dim)
    u = u_content + u_edge
    weight = torch.randn(output_dim, heads * head_dim)
    bias = torch.randn(output_dim)
    u_static = 0.7 * u

    decomposition = effective_transport_decomposition(
        u,
        node_batch,
        weight,
        output_bias=bias,
        u_content=u_content,
        u_edge=u_edge,
        u_static=u_static,
    )

    direct = torch.nn.functional.linear(u.reshape(n, -1), weight, bias)
    assert torch.allclose(decomposition["oh_reconstructed"], direct, atol=1e-6)
    assert torch.allclose(
        decomposition["u_broadcast"] + decomposition["u_residual"], u, atol=1e-7
    )
    assert torch.allclose(
        decomposition["t_content"] + decomposition["t_edge"], decomposition["t"], atol=1e-6
    )
    relational_from_parts = (
        decomposition["content_relational_energy"]
        + decomposition["edge_relational_energy"]
        + decomposition["content_edge_cross_term"]
    )
    assert torch.allclose(
        relational_from_parts, decomposition["relational_energy"], atol=2e-6
    )
    _, residual, _, _ = graphwise_broadcast_residual(decomposition["t"], node_batch)
    for graph in range(2):
        assert torch.allclose(
            residual[node_batch == graph].sum(dim=0),
            torch.zeros_like(residual[0]),
            atol=1e-6,
        )


def test_effective_treatments_are_exact_and_permutation_is_energy_matched():
    torch.manual_seed(31)
    node_batch = torch.tensor([0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    u_content = torch.randn(7, 3, 2)
    u_edge = torch.randn(7, 3, 2)
    u = u_content + u_edge
    selected = [0, 2]
    broadcast, residual, _, _ = graphwise_broadcast_residual(u, node_batch)

    only_broadcast = apply_effective_treatment(
        u, node_batch, selected, "broadcast_only"
    )
    zeroed = apply_effective_treatment(u, node_batch, selected, "full_head_zero")
    permuted = apply_effective_treatment(
        u,
        node_batch,
        selected,
        "residual_permuted",
        graph_ids=[101, 202],
        layer_index=3,
        permutation_index=1,
        seed=9,
    )
    remove_both = apply_effective_treatment(
        u,
        node_batch,
        selected,
        "remove_both_residual",
        u_content=u_content,
        u_edge=u_edge,
    )

    assert torch.allclose(only_broadcast[:, selected], broadcast[:, selected])
    assert torch.count_nonzero(zeroed[:, selected]) == 0
    assert torch.allclose(remove_both[:, selected], broadcast[:, selected], atol=1e-6)
    assert torch.equal(permuted[:, 1], u[:, 1])  # untargeted head is unchanged
    for graph in range(2):
        mask = node_batch == graph
        for head in selected:
            clean_r = residual[mask, head]
            perm_r = permuted[mask, head] - broadcast[mask, head]
            assert torch.allclose(perm_r.sum(dim=0), torch.zeros_like(perm_r[0]), atol=1e-6)
            assert torch.allclose(perm_r.pow(2).sum(), clean_r.pow(2).sum(), atol=1e-6)
            assert not torch.equal(perm_r, clean_r)


def test_paired_bootstrap_is_deterministic_and_uses_graph_mean():
    values = np.array([-1.0, 1.0, 2.0, 6.0])
    first = paired_bootstrap_summary(values, replicates=200, seed=7)
    second = paired_bootstrap_summary(values, replicates=200, seed=7)

    assert first == second
    assert first["mean"] == pytest.approx(values.mean())
    assert first["ci_low"] <= first["mean"] <= first["ci_high"]
    assert first["n_graphs"] == len(values)


def test_run_effective_transport_exercises_scoped_hooks_and_persistence_schema(monkeypatch):
    class FakeData:
        def __init__(self, **fields):
            self.__dict__.update(fields)
            self.num_nodes = int(self.x.size(0))

    class FakeBatch(FakeData):
        @classmethod
        def from_data_list(cls, graphs):
            xs, edges, edge_attrs, log_degs, batches, ys = [], [], [], [], [], []
            offset = 0
            for graph_index, graph in enumerate(graphs):
                xs.append(graph.x)
                edges.append(graph.edge_index + offset)
                edge_attrs.append(graph.edge_attr)
                log_degs.append(graph.log_deg)
                batches.append(torch.full((graph.num_nodes,), graph_index, dtype=torch.long))
                ys.append(graph.y.reshape(1, -1))
                offset += graph.num_nodes
            result = cls(
                x=torch.cat(xs),
                edge_index=torch.cat(edges, dim=1),
                edge_attr=torch.cat(edge_attrs),
                log_deg=torch.cat(log_degs),
                batch=torch.cat(batches),
                y=torch.cat(ys),
            )
            result.num_graphs = len(graphs)
            return result

        def to(self, device):
            for key, value in list(self.__dict__.items()):
                if isinstance(value, torch.Tensor):
                    setattr(self, key, value.to(device))
            return self

        def get(self, key, default=None):
            return getattr(self, key, default)

    fake_torch_geometric = ModuleType("torch_geometric")
    pyg_data = ModuleType("torch_geometric.data")
    pyg_data.Data = FakeData
    pyg_data.Batch = FakeBatch
    fake_torch_geometric.data = pyg_data
    monkeypatch.setitem(sys.modules, "torch_geometric", fake_torch_geometric)
    monkeypatch.setitem(sys.modules, "torch_geometric.data", pyg_data)

    torch.manual_seed(41)
    heads, head_dim, hidden = 2, 2, 4

    class FakeAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.num_heads = heads
            self.out_dim = head_dim
            self.edge_enhance = True
            self.V = torch.nn.Linear(hidden, hidden, bias=False)
            self.E = torch.nn.Linear(hidden, hidden, bias=False)
            self.VeRow = torch.nn.Parameter(torch.randn(head_dim, heads, head_dim) * 0.2)

        def forward(self, batch):
            src, dst = batch.edge_index
            batch.V_h = self.V(batch.x).reshape(-1, heads, head_dim)
            edge_state = torch.tanh(self.E(batch.edge_attr)).reshape(-1, heads, head_dim)
            logits = batch.V_h[src].mean(dim=-1) + edge_state.mean(dim=-1)
            attention = _normalised_attention(logits, dst, int(batch.x.size(0)))
            batch.attn = attention.unsqueeze(-1)
            value_edge = torch.einsum("ehd,dhc->ehc", edge_state, self.VeRow)
            messages = attention.unsqueeze(-1) * (batch.V_h[src] + value_edge)
            batch.wV = _scatter(messages, dst, int(batch.x.size(0)))
            batch.wE = edge_state.flatten(1)
            return batch.wV, batch.wE

    class FakeLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = FakeAttention()
            self.deg_scaler = True
            self.deg_coef = torch.nn.Parameter(torch.randn(1, hidden, 2) * 0.2)
            self.O_h = torch.nn.Linear(hidden, hidden)
            self.rezero = False

        def forward(self, batch):
            residual = batch.x
            wv, _ = self.attention(batch)
            flat = wv.reshape(batch.x.size(0), hidden)
            log_deg = batch.log_deg.reshape(-1, 1)
            flat = flat * (self.deg_coef[..., 0] + log_deg * self.deg_coef[..., 1])
            batch.x = residual + torch.tanh(self.O_h(flat))
            return batch

    class FakeCore(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([FakeLayer()])

        def forward(self, batch):
            for layer in self.layers:
                batch = layer(batch)
            return batch

    class FakeWrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = FakeCore()
            self.readout = torch.nn.Linear(hidden, 1)

        def forward(self, batch):
            batch = self.model(batch)
            pooled = batch.x.new_zeros((int(batch.num_graphs), hidden))
            pooled.index_add_(0, batch.batch, batch.x)
            return self.readout(pooled), batch.y.reshape(int(batch.num_graphs), -1)

    def graph(n, graph_seed):
        generator = torch.Generator().manual_seed(graph_seed)
        src = torch.arange(n).repeat(n)
        dst = torch.arange(n).repeat_interleave(n)
        return pyg_data.Data(
            x=torch.randn(n, hidden, generator=generator),
            edge_index=torch.stack([src, dst]),
            edge_attr=torch.randn(n * n, hidden, generator=generator),
            log_deg=torch.full((n, 1), float(np.log(n + 1.0))),
            y=torch.randn(1, generator=generator),
        )

    class FakeGm:
        def __init__(self):
            self.model = FakeWrapper().eval()
            self.eval_ds = [graph(3, 1), graph(4, 2), graph(3, 3)]
            self.device = torch.device("cpu")
            self.L, self.H, self.dh = 1, heads, head_dim
            self.loss_fun = "l1"
            self.attn_layers = [self.model.model.layers[0].attention]

        def collect_preds_ablated(self, data_groups, ablations=None):
            by_layer = {}
            for layer, head in list(ablations or []):
                by_layer.setdefault(int(layer), []).append(int(head))
            handles = []
            for layer, selected in by_layer.items():
                def hook(_module, _inputs, output, selected=tuple(selected)):
                    wv, edge = output
                    wv = wv.clone()
                    wv[:, selected, :] = 0.0
                    return wv, edge
                handles.append(self.model.model.layers[layer].attention.register_forward_hook(hook))
            predictions = []
            try:
                with torch.no_grad():
                    for data_group in data_groups:
                        batch = pyg_data.Batch.from_data_list(list(data_group))
                        pred, _ = self.model(batch)
                        predictions.append(pred.numpy())
            finally:
                for handle in handles:
                    handle.remove()
            return np.concatenate(predictions, axis=0)

    gm = FakeGm()
    result = {"gm": gm}
    config = SimpleNamespace(effective_transport_batch_size=2)
    output = run_effective_transport(
        result,
        config,
        graph_ids=[0, 1, 2],
        head_selection={"semantic": [(0, 0), (0, 1)]},
        topk=(1,),
        primary_k=1,
        residual_permutations=2,
        bootstrap_replicates=20,
        seed=5,
    )

    assert output["checks"]["passed"] is True
    assert output["checks"]["max_wv_reconstruction_error"] < 1e-5
    assert output["checks"]["max_full_zero_parity_error"] < 1e-6
    assert output["head_metrics"]["routing_js_static"].shape == (1, heads)
    assert output["head_metrics_per_graph"]["effective_total_energy"].shape == (3, 1, heads)
    causal = output["causal"]["semantic"]["1"]
    assert set(causal) == {
        "static_routing",
        "broadcast_only",
        "residual_permuted",
        "full_head_zero",
        "remove_content_residual",
        "remove_edge_residual",
    }
    assert causal["broadcast_only"]["delta_mae_per_graph"].shape == (3,)
    # All temporary hooks must be gone after the command returns.
    for module in (gm.model.model.layers[0].attention, gm.model.model.layers[0].O_h):
        assert not module._forward_hooks
        assert not module._forward_pre_hooks
