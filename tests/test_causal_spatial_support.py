from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data

from graph_specialisation_metrics.methodology.backend import CanonicalGritBackend
from graph_specialisation_metrics.methodology.causal_spatial_support import (
    Config,
    _event_conditions,
    _graph_profile_matrix,
    _load_task_context,
    _normalise_profile,
    _profile_interval,
    select_role_heads,
)
from graph_specialisation_metrics.methodology.grit_figure_data import CanonicalHeadMetrics


def test_event_conditions_cover_each_shell_and_whole_graph() -> None:
    pristine = np.asarray([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=float)
    conditions = _event_conditions({"family:semantic": ((0, 1),)}, pristine, 0)
    assert [row["distance"] for row in conditions] == [0, 1, 2, "all"]
    assert conditions[2]["nodes"] == (2,)
    assert conditions[-1]["nodes"] == (0, 1, 2)


def test_graph_profile_aggregates_donors_then_sources() -> None:
    rows = [
        {"graph": 0, "source": 0, "donor": 0, "target": "x", "distance": 0, "M": 1.0},
        {"graph": 0, "source": 0, "donor": 1, "target": "x", "distance": 0, "M": 3.0},
        {"graph": 0, "source": 1, "donor": 0, "target": "x", "distance": 0, "M": 6.0},
    ]
    matrix, graphs = _graph_profile_matrix(rows, key="M", distances=(0, 1), target="x")
    assert graphs == (0,)
    np.testing.assert_allclose(matrix, [[4.0, 0.0]])


def test_role_selection_uses_frozen_families_and_controls() -> None:
    shape = (2, 4)
    metrics = CanonicalHeadMetrics(
        raw_semantic=np.ones(shape),
        raw_structural=np.ones(shape),
        normalized_semantic=np.ones(shape),
        normalized_structural=np.ones(shape),
        joint_sensitivity=np.arange(8, dtype=float).reshape(shape),
        selectivity=np.linspace(-1, 1, 8).reshape(shape),
        active=np.ones(shape, dtype=bool),
        estimable=True,
        distance_axis=(),
        clean_attention_distance=None,
    )
    scores = {
        "families": {
            "semantic_leaning": ((1, 3), (1, 2)),
            "structural_leaning": ((0, 0), (0, 1)),
            "central_responsive": ((1, 1), (1, 0)),
        },
        "matched_controls": {
            "semantic_leaning_central_control": ((0, 3), (0, 2)),
            "structural_leaning_central_control": ((1, 0), (1, 1)),
        },
    }
    roles = select_role_heads(scores, metrics, count=1)
    assert roles["semantic"] == ((1, 3),)
    assert roles["structural"] == ((0, 0),)
    assert roles["generalist"] == ((1, 1),)
    assert roles["semantic_control"] == ((0, 3),)


class _Attention(nn.Module):
    def __init__(self, heads: int):
        super().__init__()
        self.heads = heads

    def forward(self, batch):
        routed = batch.x[:, None, None].expand(-1, self.heads, 1).clone()
        return routed, None


class _Model(nn.Module):
    def __init__(self, attention: _Attention):
        super().__init__()
        self.attention = attention

    def forward(self, batch):
        routed, _ = self.attention(batch)
        values = routed.sum(dim=(1, 2))
        prediction = torch.zeros(int(batch.num_graphs), 1, device=values.device)
        prediction[:, 0].index_add_(0, batch.batch, values)
        return prediction, prediction.clone()


def test_masked_patch_many_applies_replica_specific_nodes_and_heads() -> None:
    attention = _Attention(heads=2)
    gm = SimpleNamespace(
        L=1,
        H=2,
        dh=1,
        dim_h=2,
        device=torch.device("cpu"),
        attn_layers=[attention],
        model=_Model(attention),
    )
    output = SimpleNamespace(transform=lambda prediction, sigma: prediction)
    task = SimpleNamespace(virtual_node=False, output=output)
    backend = CanonicalGritBackend(gm, task, sigma=(1.0,))
    graph = Data(x=torch.zeros(3), edge_index=torch.empty((2, 0), dtype=torch.long))
    donor = (torch.full((6, 2, 1), 10.0),)
    _, z, _ = backend.patch_masked_many(
        [graph, graph],
        donor,
        [
            {"family": ((0, 0),), "nodes": (1,)},
            {"family": ((0, 1),), "nodes": (0, 2)},
        ],
    )
    np.testing.assert_allclose(z.detach().numpy().reshape(-1), [10.0, 20.0])


def test_normalise_profile_is_radial_allocation() -> None:
    np.testing.assert_allclose(_normalise_profile([1.0, 2.0, 1.0]), [0.25, 0.5, 0.25])


def test_empty_profile_interval_preserves_distance_axis() -> None:
    profile = _profile_interval(
        np.asarray([], dtype=float), width=4, replicates=10, seed=1
    )
    assert len(profile["mean"]) == 4
    assert np.isnan(profile["mean"]).all()


def test_profile_interval_derives_long_range_share() -> None:
    profile = _profile_interval(
        np.asarray([[1.0, 1.0, 2.0]], dtype=float),
        width=3,
        distances=(0, 1, 3),
        long_range_radius=1,
        replicates=10,
        seed=1,
    )
    assert profile["expected_distance"] == 1.75
    assert profile["long_range_share"] == 0.5


def test_task_bound_protocol_is_preferred_over_stale_root_protocol(
    tmp_path, monkeypatch
) -> None:
    from graph_specialisation_metrics.methodology import causal_spatial_support as module

    task_root = tmp_path / "zinc" / "seed_42"
    task_root.mkdir(parents=True)
    (task_root / "protocol.json").write_text('{"marker": "task"}', encoding="utf-8")
    (tmp_path / "protocol.json").write_text('{"marker": "root"}', encoding="utf-8")
    artifact = SimpleNamespace(
        value={},
        file_sha256="score-sha",
        metadata={
            "contract_fingerprint": "contract-fingerprint",
            "contract": {
                "protocol_fingerprint": "wanted",
                "checkpoint_sha256": "checkpoint-sha",
                "split_fingerprint": "split-fingerprint",
                "model_geometry": {"layers": 1, "heads": 1},
            },
        },
    )
    monkeypatch.setattr(module, "load_canonical_score_artifact", lambda *a, **k: artifact)
    monkeypatch.setattr(module, "load_canonical_model_record", lambda *a, **k: {})
    monkeypatch.setattr(module.CanonicalHeadMetrics, "from_scores", lambda *a, **k: object())
    monkeypatch.setattr(module, "select_role_heads", lambda *a, **k: {"semantic": ((0, 0),)})
    seen = []

    def protocol(record, **_kwargs):
        seen.append(record["marker"])
        return SimpleNamespace(
            fingerprint="wanted" if record["marker"] == "task" else "stale"
        )

    monkeypatch.setattr(module, "methodology_config_from_record", protocol)
    context = _load_task_context(
        Config(canonical_root=tmp_path, output_dir=tmp_path / "output", tasks=("zinc",)),
        "zinc",
    )
    assert context["protocol_config"].fingerprint == "wanted"
    assert seen == ["task"]


def test_protocol_task_subset_is_exactly_reconstructed(tmp_path, monkeypatch) -> None:
    from graph_specialisation_metrics.methodology import causal_spatial_support as module

    canonical_root = tmp_path / "canonical"
    canonical_root.mkdir()
    (canonical_root / "protocol.json").write_text(
        json.dumps(
            {
                "tasks": ["zinc", "qm9_gap_dense", "peptides_func", "peptides_struct"],
                "task_train_seeds": {},
                "task_overrides": {},
            }
        ),
        encoding="utf-8",
    )

    def protocol(record, **_kwargs):
        tasks = tuple(record["tasks"])
        fingerprint = "wanted" if tasks == ("zinc", "qm9_gap_dense") else "stale"
        return SimpleNamespace(fingerprint=fingerprint, tasks=tasks)

    monkeypatch.setattr(module, "methodology_config_from_record", protocol)
    resolved, source = module._resolve_protocol_config(
        Config(canonical_root=canonical_root, output_dir=tmp_path / "output"),
        task="zinc",
        task_root=canonical_root / "zinc" / "seed_42",
        expected_fingerprint="wanted",
    )
    assert resolved.tasks == ("zinc", "qm9_gap_dense")
    assert source.endswith("#tasks=zinc,qm9_gap_dense")
