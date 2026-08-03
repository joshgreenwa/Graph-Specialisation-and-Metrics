from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from graph_specialisation_metrics.methodology.backend import CanonicalGritBackend
from graph_specialisation_metrics.methodology.grit_causal_population import (
    DENSE_GRIT_TASKS,
    GRIT_POPULATION_CAUSAL_VERSION,
    production_config,
)


def test_dense_grit_population_config_uses_disjoint_paper_populations(tmp_path):
    config = production_config(
        output_dir=str(tmp_path),
        accelerator="cpu",
        graphs_per_batch=8,
    )
    config.validate()
    assert config.tasks == DENSE_GRIT_TASKS
    assert config.seeds_for("zinc") == (42,)
    assert config.seeds_for("qm9_gap_dense") == (42,)
    assert config.sizes.discovery_graphs == 256
    assert config.sizes.causal_graphs == 256
    assert config.sizes.clean_ablation_graphs == 256
    assert config.sizes.semantic_donor_graphs == 2_000
    assert config.execution.graphs_per_batch == 8


def test_dense_grit_population_config_rejects_sparse_controls(tmp_path):
    with pytest.raises(ValueError, match="dense causal population tasks"):
        production_config(
            output_dir=str(tmp_path),
            tasks=("zinc_1hop",),
            accelerator="cpu",
        )


def test_canonical_grit_backend_assigns_heads_by_graph_with_variable_node_counts():
    torch = pytest.importorskip("torch")
    pyg_data = pytest.importorskip("torch_geometric.data")

    class RoutedLayer(torch.nn.Module):
        def forward(self, routed):
            return routed, None

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList((RoutedLayer(), RoutedLayer()))

        def forward(self, batch):
            routed = batch.x.reshape(-1, 2, 1).float()
            for layer in self.layers:
                routed, _edge = layer(routed)
            prediction = routed.new_zeros((int(batch.num_graphs), 1))
            prediction.index_add_(0, batch.batch, routed.sum(dim=(1, 2), keepdim=False)[:, None])
            return prediction, prediction.new_zeros(prediction.shape)

    class IdentityOutput:
        @staticmethod
        def transform(prediction, sigma):
            del sigma
            return prediction

    model = Model()
    runtime = SimpleNamespace(
        L=2,
        H=2,
        dh=1,
        dim_h=2,
        device=torch.device("cpu"),
        model=model,
        attn_layers=model.layers,
    )
    task = SimpleNamespace(output=IdentityOutput(), virtual_node=False)
    backend = CanonicalGritBackend(runtime, task, sigma=(1.0,))
    graphs = (
        pyg_data.Data(x=torch.tensor([[1.0, 2.0], [3.0, 4.0]])),
        pyg_data.Data(x=torch.tensor([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])),
    )
    assignments = ((0, 0), (1, 1))

    _prediction, ablated, _target = backend.ablate_individual_heads(
        graphs,
        assignments,
    )
    assert np.allclose(ablated.detach().numpy().reshape(-1), (6.0, 90.0))

    replacements = tuple(torch.full((5, 2, 1), value) for value in (100.0, -5.0))
    _prediction, patched, _target = backend.patch_individual_heads(
        graphs,
        replacements,
        assignments,
    )
    assert np.allclose(patched.detach().numpy().reshape(-1), (206.0, 75.0))
    assert GRIT_POPULATION_CAUSAL_VERSION.startswith("grit-dense-")
