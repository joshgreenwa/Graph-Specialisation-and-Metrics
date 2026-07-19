import copy
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from graph_specialisation_metrics.specialisation.causal_mediation import (
    calibrate_head_scores,
    directional_mediation_components,
    make_head_group_specs,
    ratio_of_sums,
    run_channel_causal_mediation,
    select_confirmation_graph_ids,
)


def test_calibration_is_channel_scale_invariant_and_filters_inert_heads():
    sem = np.array([[8.0, 2.0, 0.01, 0.01], [6.0, 2.0, 0.01, 0.01]])
    rrwp = np.array([[1.0, 2.0, 0.01, 7.0], [1.0, 2.0, 0.01, 6.0]])
    first = calibrate_head_scores(sem, rrwp, importance_quantile=0.5, min_eligible=4)
    scaled = calibrate_head_scores(sem * 17.0, rrwp * 0.2, importance_quantile=0.5, min_eligible=4)

    assert np.allclose(first["importance"], scaled["importance"])
    assert np.allclose(first["preference"], scaled["preference"])
    assert not bool(first["eligible"][0, 2])
    assert first["rankings"]["semantic"][0][1] == 0
    assert first["rankings"]["rrwp_role"][0][1] == 3


def test_confirmation_selection_is_deterministic_and_disjoint():
    discovery = np.array([0, 2, 4, 6])
    first = select_confirmation_graph_ids(20, discovery, 7, seed=9)
    second = select_confirmation_graph_ids(20, discovery, 7, seed=9)

    assert np.array_equal(first, second)
    assert len(first) == 7
    assert np.intersect1d(first, discovery).size == 0


def test_group_specs_are_disjoint_and_controls_match_layers_when_available():
    sem = np.array([[9.0, 6.0, 2.0, 1.0, 0.8, 0.7], [8.0, 5.0, 2.0, 1.0, 0.8, 0.7]])
    rrwp = np.array([[0.7, 0.8, 1.0, 2.0, 5.0, 9.0], [0.7, 0.8, 1.0, 2.0, 6.0, 8.0]])
    calibration = calibrate_head_scores(sem, rrwp, min_eligible=8)
    specs = make_head_group_specs(
        calibration, topk=(1, 2, 4), matched_control_draws=2, seed=3
    )

    for k in (1, 2, 4):
        sem_group = next(row for row in specs if row["group_id"] == f"semantic_top{k}")
        rrwp_group = next(row for row in specs if row["group_id"] == f"rrwp_role_top{k}")
        sem_heads = {tuple(h) for h in sem_group["heads"]}
        rrwp_heads = {tuple(h) for h in rrwp_group["heads"]}
        assert not (sem_heads & rrwp_heads)
    controls = [row for row in specs if row["group_type"] == "matched_control"]
    assert controls
    assert all(len(row["heads"]) == row["k"] for row in controls)
    assert all(0.0 <= row["match_quality"]["exact_layer_fraction"] <= 1.0 for row in controls)
    specialised_extremes = {
        tuple(head)
        for row in specs
        if row["group_type"] == "specialised" and row["k"] == 4
        for head in row["heads"]
    }
    assert all(not (set(map(tuple, row["heads"])) & specialised_extremes) for row in controls)


def test_directional_components_validate_shapes_and_ratio_is_ratio_of_sums():
    clean = np.array([[0.0], [0.0]])
    cf = np.array([[1.0], [100.0]])
    patch = np.array([[[1.0]], [[1.0]]])
    numerator, denominator, _ = directional_mediation_components(
        clean, cf, patch, direction="noising"
    )

    assert float(ratio_of_sums(numerator, denominator)[0]) == pytest.approx(101.0 / 10001.0)
    with pytest.raises(ValueError, match=r"equal \[N,T\]"):
        directional_mediation_components(clean[:, 0], cf, patch, direction="noising")
    with pytest.raises(ValueError, match=r"\[N,K,T\]"):
        directional_mediation_components(clean, cf, np.zeros((3, 2, 1)), direction="noising")


class FakeData:
    def __init__(self, x, rrwp, edge_index, edge_attr, y):
        self.x = x
        self.rrwp = rrwp
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.y = y
        self.num_nodes = int(x.shape[0])

    def clone(self):
        out = copy.copy(self)
        for name, value in vars(self).items():
            setattr(out, name, value.clone() if torch.is_tensor(value) else copy.deepcopy(value))
        return out


class FakeBatch:
    @classmethod
    def from_data_list(cls, rows):
        out = cls()
        out.x = torch.cat([row.x.clone() for row in rows], dim=0)
        out.rrwp = torch.cat([row.rrwp.clone() for row in rows], dim=0)
        out.y = torch.stack([row.y.reshape(-1).clone() for row in rows], dim=0)
        counts = [int(row.num_nodes) for row in rows]
        out.graph_num_nodes = torch.tensor(counts, dtype=torch.long)
        out.batch = torch.repeat_interleave(
            torch.arange(len(rows), dtype=torch.long), torch.tensor(counts, dtype=torch.long)
        )
        out.num_nodes = int(sum(counts))
        return out

    def to(self, device):
        for name, value in list(vars(self).items()):
            if torch.is_tensor(value):
                setattr(self, name, value.to(device))
        return self


class FakeContentAdapter:
    def rows(self, data):
        return data.x.detach().cpu().numpy().astype(np.int64)

    def write_donors(self, batch_x, row_idx, donor_rows):
        batch_x[row_idx] = torch.as_tensor(donor_rows, dtype=batch_x.dtype, device=batch_x.device)


class FakeAttention(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = int(layer)

    def forward(self, batch):
        x = batch.x.float().reshape(-1)
        role = batch.rrwp[:, 0].float()
        if self.layer == 0:
            sem = 2.0 * x
            mixed = 0.10 * x * role
            rrwp = 1.20 * x * role
        else:
            sem = 1.25 * x
            mixed = 0.05 * x * role
            rrwp = 0.80 * x * role
        h_out = torch.stack([sem, mixed, rrwp], dim=1).unsqueeze(-1)
        return h_out, None


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.attentions = nn.ModuleList([FakeAttention(0), FakeAttention(1)])

    def forward(self, batch):
        node_signal = torch.zeros(batch.num_nodes, dtype=torch.float32, device=batch.x.device)
        for attention in self.attentions:
            h_out, _ = attention(batch)
            node_signal = node_signal + h_out.sum(dim=(1, 2))
        pred = torch.zeros(
            len(batch.graph_num_nodes), 1, dtype=torch.float32, device=batch.x.device
        )
        pred[:, 0].index_add_(0, batch.batch, node_signal)
        # Deliberately mutate the runtime batch. Reusing a Batch would make later predictions fail;
        # the mediation runner must construct a fresh Batch for every capture/patch forward.
        batch.x = batch.x + 10_000
        return pred, batch.y


class FakeGM:
    def __init__(self, dataset):
        self.eval_ds = dataset
        self.donor_ds = dataset
        self.adapter = FakeContentAdapter()
        self.device = torch.device("cpu")
        self.L = 2
        self.H = 3
        self.dh = 1
        self.loss_fun = "l1"
        self.model = FakeModel()
        self.attn_layers = list(self.model.attentions)

    def capture(self, batch, *, want_grad, want_attn=False):
        del want_grad, want_attn
        records = [None] * self.L

        def make_hook(layer):
            def hook(_module, _inputs, output):
                records[layer] = output[0]

            return hook

        handles = [
            module.register_forward_hook(make_hook(i))
            for i, module in enumerate(self.attn_layers)
        ]
        try:
            pred, true = self.model(batch)
        finally:
            for handle in handles:
                handle.remove()
        return {"pred": pred, "true": true, "wV": records, "attn": None, "edge_index": None}


def make_fake_dataset(num_graphs=10):
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    edge_attr = torch.ones(4, 1, dtype=torch.long)
    dataset = []
    for graph_id in range(num_graphs):
        values = torch.tensor(
            [[1 + graph_id % 4], [3 + (graph_id + 1) % 4], [7 + graph_id % 3]],
            dtype=torch.long,
        )
        rrwp = torch.tensor([[0.0], [1.0], [3.0]])
        dataset.append(
            FakeData(values, rrwp, edge_index.clone(), edge_attr.clone(), torch.tensor([0.0]))
        )
    return dataset


@pytest.fixture
def fake_pyg(monkeypatch):
    package = types.ModuleType("torch_geometric")
    data_module = types.ModuleType("torch_geometric.data")
    data_module.Batch = FakeBatch
    package.data = data_module
    monkeypatch.setitem(sys.modules, "torch_geometric", package)
    monkeypatch.setitem(sys.modules, "torch_geometric.data", data_module)


def test_full_runner_uses_heldout_frozen_bank_and_returns_persistence_schema(fake_pyg):
    gm = FakeGM(make_fake_dataset())
    result = {
        "task": "zinc",
        "title": "fake ZINC",
        "gm": gm,
        "graph_ids": np.array([0, 1]),
        "S_sem": np.array([[9.0, 2.0, 0.5], [8.0, 2.0, 0.5]]),
        "S_str": np.array([[0.5, 2.0, 9.0], [0.5, 2.0, 8.0]]),
    }
    sc = SimpleNamespace(
        donor_split="test",
        eval_split="test",
        partner_match="degree",
        tol=1.0e-5,
        float_noise_tol=1.0e-4,
    )

    out = run_channel_causal_mediation(
        result,
        sc,
        n_graphs=4,
        interventions_per_graph=1,
        topk=(1,),
        primary_k=1,
        matched_control_draws=1,
        bootstrap_replicates=20,
        seed=7,
    )

    assert out["schema_version"] == 1
    assert np.intersect1d(out["discovery_graph_ids"], out["confirmation_graph_ids"]).size == 0
    # Four graphs x one paired intervention x two channels.
    assert len(out["intervention_bank"]) == 8
    assert {row["channel"] for row in out["intervention_bank"]} == {"semantic", "rrwp_role"}
    by_pair = {}
    for row in out["intervention_bank"]:
        by_pair.setdefault(row["pair_id"], []).append(row)
    assert all(
        len(rows) == 2 and rows[0]["anchor"] == rows[1]["anchor"]
        for rows in by_pair.values()
    )

    # Six heads, two channels and two directions; every head was causally patched.
    assert len(out["single_head"]["summary"]) == 6 * 2 * 2
    assert out["single_head"]["patched_predictions"]["noising"].shape == (8, 6, 1)
    assert out["contrasts"]["topk"]
    assert out["contrasts"]["interactions"]
    assert out["contrasts"]["continuous"][0]["permutations"] == 20
    assert 0.0 <= out["contrasts"]["continuous"][0]["permutation_p_two_sided"] <= 1.0
    assert out["checks"]["discovery_confirmation_overlap"] == 0
    assert out["checks"]["same_source_clean_patch_max_abs"] <= out["checks"]["noop_tolerance"]
    assert out["checks"]["semantic_total_effect_energy"] > 0
    assert out["checks"]["rrwp_role_total_effect_energy"] > 0

    sem_heads = set(map(tuple, out["calibration"]["rankings"]["semantic"][:1]))
    rrwp_spec = next(row for row in out["groups"]["specs"] if row["group_id"] == "rrwp_role_top1")
    assert not (sem_heads & set(map(tuple, rrwp_spec["heads"])))
    # All temporary hooks, including hooks from exceptional-safe patch contexts, were removed.
    assert all(not module._forward_hooks for module in gm.attn_layers)
