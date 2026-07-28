from __future__ import annotations

import ast
import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from graph_specialisation_metrics.methodology.grit_figure_data import (  # noqa: E402
    CanonicalHeadMetrics,
    GritDiagnosticExtractor,
    SupplementalCache,
    load_canonical_model_record,
    load_canonical_score_artifact,
    methodology_config_from_record,
    select_ranked_heads,
    select_specialist_heads,
    select_structural_specialist_head,
)
from graph_specialisation_metrics.methodology.grit_figure_plots import (  # noqa: E402
    plot_attention_grid,
    plot_av_pca,
    plot_coordinate_heatmaps,
    plot_logit_spread,
    plot_score_heatmaps,
    plot_score_plane,
    plot_selectivity_joint_plane,
    plot_selectivity_vs_logit_ratio,
)
from graph_specialisation_metrics.methodology.protocol import (  # noqa: E402
    ExecutionPolicy,
    MethodologyConfig,
    PROTOCOL_VERSION,
    SplitManifest,
    stable_hash,
)
from graph_specialisation_metrics.methodology import runner  # noqa: E402


def _score_value():
    selectivity = np.asarray(
        [[0.80, -0.70, 0.05, 0.01], [0.60, -0.40, 0.08, -0.02]]
    )
    joint = np.asarray(
        [[1.30, 1.10, 2.20, 1.80], [1.20, 1.00, 2.00, 1.70]]
    )
    semantic = joint * (1.0 + selectivity)
    structural = joint * (1.0 - selectivity)
    coordinates = SimpleNamespace(
        raw_semantic=semantic,
        raw_structural=structural,
        normalized_semantic=semantic,
        normalized_structural=structural,
        joint_sensitivity=joint,
        selectivity=selectivity,
        active=np.ones_like(selectivity, dtype=bool),
        estimable=True,
    )
    return {
        "coordinates": coordinates,
        "channels": {
            "semantic": {"raw": semantic},
            "structural": {"raw": structural},
        },
        "axis": ("0", "1", "2"),
        "clean_attention_distance": np.full((2, 4, 3), 1.0 / 3.0),
    }


def _contract(splits: SplitManifest) -> dict:
    return {
        "protocol_fingerprint": "protocol",
        "task": "zinc",
        "task_adapter_version": "canonical-grit-v1",
        "checkpoint_sha256": "checkpoint",
        "train_seed": 42,
        "model_geometry": {
            "layers": 2,
            "heads": 4,
            "head_width": 8,
            "hidden_width": 32,
            "outputs": 1,
        },
        "output_representation": "evaluation_regression",
        "sigma": (1.0,),
        "split_fingerprint": splits.fingerprint,
        "event_manifest_hash": "events",
        "donors_per_source": 8,
        "source_cap": 6,
        "bootstrap_seed": 17_071,
    }


def test_head_metrics_and_grit_specialist_selection():
    metrics = CanonicalHeadMetrics.from_scores(_score_value())
    assert metrics.shape == (2, 4)
    selected = select_specialist_heads(metrics)
    assert selected == {"semantic": (0, 0), "structural": (0, 1)}
    assert select_structural_specialist_head(
        metrics, excluded_heads=((0, 1),)
    ) == (1, 1)
    ranked = select_ranked_heads(
        metrics,
        semantic_count=3,
        joint_count=3,
        joint_generalist_max_abs_selectivity=0.10,
    )
    assert ranked["top_semantic_1"] == (0, 0)
    assert ranked["top_joint_1"] == (0, 2)


def test_score_and_model_artifacts_are_bound_to_one_task(tmp_path: Path):
    torch = pytest.importorskip("torch")
    splits = SplitManifest(
        discovery=(0,),
        causal=(1,),
        clean_ablation=(2,),
        semantic_donor_pool=(3, 4),
        same_index_space=False,
        seed=31_415,
    )
    contract = _contract(splits)
    cache_path = tmp_path / "raw.pt"
    torch.save(
        {
            "metadata": {
                "protocol_version": PROTOCOL_VERSION,
                "contract": contract,
                "contract_fingerprint": stable_hash(contract),
            },
            "value": _score_value(),
        },
        cache_path,
    )
    artifact = load_canonical_score_artifact(
        cache_path, expected_task="zinc"
    )
    assert artifact.metadata["contract"]["task"] == "zinc"
    with pytest.raises(Exception, match="not 'qm9_gap_dense'"):
        load_canonical_score_artifact(
            cache_path, expected_task="qm9_gap_dense"
        )

    model_path = tmp_path / "model.json"
    model_path.write_text(
        json.dumps(
            {
                "task": "zinc",
                "task_adapter_version": "canonical-grit-v1",
                "checkpoint_sha256": "checkpoint",
                "train_seed": 42,
                "splits": dataclasses.asdict(splits),
            }
        ),
        encoding="utf-8",
    )
    record = load_canonical_model_record(model_path, artifact)
    assert record["splits"]["seed"] == 31_415


def test_supplemental_cache_is_contract_keyed(tmp_path: Path):
    calls = []
    cache = SupplementalCache(tmp_path)
    first, first_path, first_hit = cache.load_or_compute(
        "diagnostic",
        {"task": "zinc", "head": [0, 1]},
        lambda: calls.append("zinc") or {"value": 1},
    )
    second, second_path, second_hit = cache.load_or_compute(
        "diagnostic",
        {"task": "zinc", "head": [0, 1]},
        lambda: calls.append("unexpected") or {"value": 2},
    )
    _, qm9_path, qm9_hit = cache.load_or_compute(
        "diagnostic",
        {"task": "qm9_gap_dense", "head": [0, 1]},
        lambda: calls.append("qm9") or {"value": 3},
    )
    assert first == second == {"value": 1}
    assert first_path == second_path
    assert not first_hit and second_hit and not qm9_hit
    assert qm9_path != first_path
    assert calls == ["zinc", "qm9"]


def test_graph_local_estimator_normalises_each_graph_independently(
    monkeypatch,
):
    semantic = {
        10: np.asarray([[4.0, 1.0], [1.0, 2.0]]),
        20: np.asarray([[1.0, 4.0], [2.0, 1.0]]),
    }
    structural = {
        10: np.asarray([[1.0, 2.0], [4.0, 1.0]]),
        20: np.asarray([[3.0, 1.0], [1.0, 3.0]]),
    }

    def fake_score_batch(
        _prepared,
        _config,
        _plan,
        _clean,
        axis,
        channel,
        graph_ids,
    ):
        assert axis == "display-axis"
        values = semantic if channel == "semantic" else structural
        return [
            {"graph_id": int(graph_id), "score": values[int(graph_id)]}
            for graph_id in graph_ids
        ]

    monkeypatch.setattr(runner, "_score_graph_batch", fake_score_batch)
    monkeypatch.setattr(runner, "_event_item_cost", lambda *args, **kwargs: 1)
    monkeypatch.setattr(
        runner, "_distance_axis", lambda _prepared, _graph_ids: "display-axis"
    )
    backend = SimpleNamespace(
        clean_jacobians_many=lambda bases: [
            f"clean-{base}" for base in bases
        ]
    )
    runtime = SimpleNamespace(
        eval_ds={10: "graph-10", 20: "graph-20"},
        sc=SimpleNamespace(seed=0),
    )
    prepared = SimpleNamespace(
        task=SimpleNamespace(name="synthetic"),
        runtime=runtime,
        grit=runtime,
        backend=backend,
    )
    plan = {
        graph_id: {
            channel: {"records": [], "sources": ()}
            for channel in ("semantic", "structural")
        }
        for graph_id in (10, 20)
    }
    config = MethodologyConfig(
        tasks=("synthetic",),
        train_seeds=(0,),
        execution=ExecutionPolicy(graphs_per_batch=2),
    )
    result = runner.estimate_graph_local_head_coordinates(
        prepared,
        config,
        plan=plan,
    )
    first, second = result["graphs"][10], result["graphs"][20]
    assert np.isclose(first["normalized_semantic"].mean(), 1.0)
    assert np.isclose(first["normalized_structural"].mean(), 1.0)
    assert np.isclose(second["normalized_semantic"].mean(), 1.0)
    assert np.isclose(second["normalized_structural"].mean(), 1.0)
    assert not np.allclose(first["selectivity"], second["selectivity"])
    assert result["graph_id_seed_space"] == "caller-supplied graph ID"


def test_protocol_record_reconstructs_runtime_configuration(tmp_path: Path):
    original = MethodologyConfig(
        output_dir=str(tmp_path / "canonical"),
        tasks=("zinc", "qm9_gap_dense"),
        train_seeds=(42,),
        phases=("scores",),
    )
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(original.record()), encoding="utf-8")
    restored = methodology_config_from_record(path, accelerator="cpu")
    assert restored.output_dir == original.output_dir
    assert restored.tasks == original.tasks
    assert restored.sizes == original.sizes
    assert restored.families == original.families
    assert restored.phases == ()
    assert restored.accelerator == "cpu"
    assert restored.fingerprint == original.fingerprint


def test_grit_plotting_api_accepts_synthetic_payloads():
    import matplotlib.pyplot as plt

    metrics = CanonicalHeadMetrics.from_scores(_score_value())
    selected = {"semantic": (0, 0), "structural": (0, 1)}
    figures = [
        plot_score_heatmaps(metrics, selected, title="Synthetic GRIT"),
        plot_coordinate_heatmaps(metrics, selected, title="Synthetic GRIT"),
        plot_score_plane(metrics, selected, title="Synthetic GRIT"),
        plot_selectivity_joint_plane(
            metrics, selected, title="Synthetic GRIT"
        ),
    ]
    attention = {
        "task": "zinc",
        "examples": [
            {
                "dataset_index": 0,
                "n_atoms": 3,
                "node_labels": ["C", "O", "N"],
                "edge_index": np.asarray(
                    [[0, 1, 1, 2], [1, 0, 2, 1]], dtype=np.int64
                ),
                "attention": {
                    "semantic": np.asarray(
                        [
                            [0.6, 0.4, 0.0],
                            [0.2, 0.5, 0.3],
                            [0.0, 0.4, 0.6],
                        ]
                    )
                },
            }
        ],
    }
    figures.append(
        plot_attention_grid(
            attention,
            role="semantic",
            head=(0, 0),
            per_graph_coordinates={0: {"D_rel": 0.8, "J": 1.3}},
            net_d_rel=0.8,
            net_joint_sensitivity=1.3,
        )
    )
    figures.append(
        plot_av_pca(
            {
                "task": "zinc",
                "head": (0, 0),
                "vectors": np.arange(24, dtype=float).reshape(8, 3),
                "labels": ["C-focused"] * 4 + ["O-focused"] * 4,
                "n_used": 8,
            }
        )
    )
    logit = {
        "task": "zinc",
        "n_used": 5,
        "node_std_mean": np.asarray([0.5, 0.8]),
        "node_std_std": np.asarray([0.1, 0.1]),
        "relation_std_mean": np.asarray([0.7, 0.9]),
        "relation_std_std": np.asarray([0.1, 0.2]),
        "log_r_mean": np.zeros((2, 4)),
    }
    figures.append(plot_logit_spread(logit))
    figures.append(
        plot_selectivity_vs_logit_ratio(metrics, logit, selected)
    )
    assert all(figure.axes for figure in figures)
    for figure in figures:
        plt.close(figure)


def test_grit_diagnostic_extractor_captures_native_sparse_sites():
    torch = pytest.importorskip("torch")
    pyg_data = pytest.importorskip("torch_geometric.data")
    pyg_utils = pytest.importorskip("torch_geometric.utils")

    class FakeAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.num_heads = 2
            self.out_dim = 2
            self.clamp = 5.0
            self.act = torch.nn.ReLU()
            self.Aw = torch.nn.Parameter(
                torch.asarray(
                    [
                        [[0.8], [0.6]],
                        [[0.3], [0.5]],
                    ]
                )
            )

        def forward(self, batch):
            batch.Q_h = batch.x.view(-1, self.num_heads, self.out_dim)
            batch.K_h = (0.5 * batch.x).view(
                -1, self.num_heads, self.out_dim
            )
            batch.V_h = (0.25 * batch.x).view(
                -1, self.num_heads, self.out_dim
            )
            source, receiver = batch.edge_index
            node = batch.K_h[source] + batch.Q_h[receiver]
            relation = self.act(2.0 * node + 0.1)
            batch.wE = relation.flatten(1)
            raw = torch.einsum("ehd,dhc->ehc", relation, self.Aw)
            batch.attn = pyg_utils.softmax(raw, receiver)
            message = batch.V_h[source] * batch.attn
            batch.wV = torch.zeros_like(batch.V_h)
            batch.wV.index_add_(0, receiver, message)
            return batch.wV, batch.wE

    class FakeModel(torch.nn.Module):
        def __init__(self, attention):
            super().__init__()
            self.attention = attention

        def forward(self, batch):
            self.attention(batch)
            return torch.zeros((1, 1)), torch.zeros((1, 1))

    attention = FakeAttention()
    grit = SimpleNamespace(
        attn_layers=[attention],
        model=FakeModel(attention),
        device=torch.device("cpu"),
        L=1,
        H=2,
    )
    runtime = SimpleNamespace(
        runtime=grit,
        prepared=SimpleNamespace(task=SimpleNamespace(name="zinc")),
    )
    graph = pyg_data.Data(
        x=torch.arange(12, dtype=torch.float32).reshape(3, 4) / 10,
        edge_index=torch.asarray(
            [
                [0, 1, 2, 0, 1, 2, 0, 1, 2],
                [0, 0, 0, 1, 1, 1, 2, 2, 2],
            ],
            dtype=torch.long,
        ),
    )
    captured = GritDiagnosticExtractor(runtime).extract(graph)
    assert captured.attention[0].shape == (2, 3, 3)
    assert captured.transport[0].shape == (3, 2, 2)
    np.testing.assert_allclose(
        captured.attention[0].sum(dim=-1).numpy(),
        np.ones((2, 3)),
        atol=1e-6,
    )
    assert not torch.allclose(
        captured.node_only_logits[0],
        captured.relation_logits[0],
        equal_nan=True,
    )


def test_colab_notebook_has_valid_python_cells():
    notebook = (
        Path(__file__).parents[1]
        / "experiments"
        / "methodology"
        / "grit_zinc_qm9_figures_colab.ipynb"
    )
    payload = json.loads(notebook.read_text(encoding="utf-8"))
    assert payload["nbformat"] == 4
    assert "zinc" in "".join(payload["cells"][1]["source"])
    assert "qm9_gap_dense" in "".join(payload["cells"][1]["source"])
    for index, cell in enumerate(payload["cells"]):
        if cell["cell_type"] == "code":
            ast.parse(
                "".join(cell["source"]),
                filename=f"{notebook}:cell_{index}",
            )
