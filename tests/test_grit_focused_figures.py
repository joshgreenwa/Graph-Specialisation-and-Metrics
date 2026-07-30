from __future__ import annotations

import ast
import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib
from matplotlib.collections import PolyCollection
import numpy as np
import pytest

matplotlib.use("Agg")

from graph_specialisation_metrics.methodology.grit_figure_data import (  # noqa: E402
    CHEMISTRY_FOCUS_VERSION,
    CanonicalHeadMetrics,
    GritDiagnosticExtractor,
    SupplementalCache,
    ZINC_ATOM_TYPES,
    _normalised_attention_entropy,
    atom_chemistry_categories,
    compute_layer_av_pca_inputs,
    compute_selected_head_transport_profiles,
    figure_identity,
    graph_node_labels,
    label_attention_focus,
    load_canonical_model_record,
    load_canonical_score_artifact,
    methodology_config_from_record,
    molecule_from_graph,
    molecule_record,
    select_ranked_heads,
    select_specialist_heads,
    select_structural_specialist_head,
    select_structurally_selective_heads,
)
from graph_specialisation_metrics.methodology.grit_figure_plots import (  # noqa: E402
    MOLECULE_DRAW_DPI,
    MOLECULE_RENDER_DPI,
    PCA_FOCUS_COLORS,
    PUBLICATION_PDF_RASTER_DPI,
    PUBLICATION_PNG_DPI,
    _pca_focus_color,
    plot_attention_grid,
    plot_av_pca,
    plot_coordinate_heatmaps,
    plot_hop_attention_mass,
    plot_joint_sensitivity_vs_attention_entropy,
    plot_layer_av_pca_grid,
    plot_logit_spread,
    plot_routing_transport_profiles,
    plot_score_heatmaps,
    plot_score_plane,
    plot_selectivity_joint_plane,
    plot_selectivity_vs_attention_entropy,
    plot_selectivity_vs_logit_ratio,
    save_figure_bundle,
    save_section_pdf_bundles,
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


def _transport_score_value():
    scores = _score_value()
    metrics = CanonicalHeadMetrics.from_scores(scores)
    fractions = np.asarray([0.20, 0.30, 0.50])
    channels = {}
    for channel, raw in (
        ("semantic", metrics.raw_semantic),
        ("structural", metrics.raw_structural),
    ):
        profile = raw[..., None] * fractions
        graph_contribution = {
            10: profile * 0.8,
            20: profile * 1.2,
        }
        channels[channel] = {
            "raw": raw,
            "graph_distance_contribution": graph_contribution,
            "distance_support": {
                "axis": ("0", "1", "2"),
                "reportable": np.ones(3, dtype=bool),
            },
            "events": [
                {
                    "graph_id": graph_id,
                    "source": 0,
                    "draw": 0,
                    "distance_contribution": contribution,
                }
                for graph_id, contribution in graph_contribution.items()
            ],
            "resample_source": channel == "structural",
        }
    scores["channels"] = channels
    return scores


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
    assert select_structurally_selective_heads(metrics) == (
        (0, 1),
        (1, 1),
        (1, 3),
    )
    ranked = select_ranked_heads(
        metrics,
        semantic_count=3,
        structural_count=3,
        joint_count=3,
        joint_generalist_max_abs_selectivity=0.10,
    )
    assert ranked["top_semantic_1"] == (0, 0)
    assert ranked["top_structural_1"] == (0, 1)
    assert ranked["top_joint_1"] == (0, 2)


def test_normalised_attention_entropy_has_zero_and_uniform_endpoints():
    torch = pytest.importorskip("torch")
    attention = torch.asarray(
        [
            [[0.5, 0.5], [0.5, 0.5]],
            [[1.0, 0.0], [0.0, 1.0]],
        ]
    )
    entropy = _normalised_attention_entropy(attention)
    torch.testing.assert_close(entropy, torch.asarray([1.0, 0.0]))


def test_selected_head_transport_profiles_reconstruct_normalised_scores():
    scores = _transport_score_value()
    heads = ((0, 0), (1, 2))
    payload = compute_selected_head_transport_profiles(
        scores,
        heads,
        bootstrap=False,
    )

    assert payload["heads"] == heads
    assert payload["n_graphs"] == 2
    assert payload["graph_ids"] == (10, 20)
    for channel in ("semantic", "structural"):
        result = payload["channels"][channel]
        raw = np.asarray(scores["channels"][channel]["raw"])
        expected = raw / raw.mean()
        assert result["replicates"] == 0
        assert np.all(result["reportable"])
        assert np.allclose(
            result["estimate"].sum(axis=-1),
            [expected[head] for head in heads],
        )
        assert np.allclose(result["low"], result["estimate"])
        assert np.allclose(result["high"], result["estimate"])


def test_selected_head_transport_profiles_use_nested_event_bootstrap(
    monkeypatch,
):
    scores = _transport_score_value()
    heads = ((0, 0), (1, 2))
    calls = []

    def fake_interval(observations, policy, *, graph_reduce):
        calls.append((observations, policy))
        estimate = graph_reduce(
            np.stack(
                [observation.value for observation in observations]
            )
        )
        return SimpleNamespace(
            estimate=estimate,
            low=estimate * 0.9,
            high=estimate * 1.1,
            replicates=2000,
            rng_seed=17071,
            resampled_levels=("graph", "source", "donor"),
        )

    monkeypatch.setattr(
        "graph_specialisation_metrics.methodology.grit_figure_data."
        "nested_percentile_interval",
        fake_interval,
    )
    payload = compute_selected_head_transport_profiles(scores, heads)

    assert len(calls) == 2
    assert all(len(observations) == 2 for observations, _ in calls)
    assert calls[0][1].resample_source is False
    assert calls[1][1].resample_source is True
    assert all(
        payload["channels"][channel]["replicates"] == 2000
        for channel in ("semantic", "structural")
    )


def test_routing_transport_figure_facets_requested_payload_subset():
    import matplotlib.pyplot as plt

    scores = _transport_score_value()
    metrics = CanonicalHeadMetrics.from_scores(scores)
    payload_heads = (
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 0),
        (1, 2),
    )
    displayed_heads = ((0, 1), (0, 3), (1, 0), (1, 2))
    payload = compute_selected_head_transport_profiles(
        scores,
        payload_heads,
        bootstrap=False,
    )
    figure = plot_routing_transport_profiles(
        metrics,
        payload,
        heads=displayed_heads,
        title="ZINC — GRIT — routing geometry and transport response",
        dataset_label="ZINC discovery set",
    )
    try:
        assert np.allclose(figure.get_size_inches(), (17.0, 8.5))
        assert len(figure.axes) == 8
        assert [
            axis.get_title().splitlines()[0] for axis in figure.axes[:4]
        ] == ["L0 H1", "L0 H3", "L1 H0", "L1 H2"]
        assert figure.axes[0].get_ylabel() == "Mean clean attention mass"
        assert figure.axes[4].get_ylabel() == (
            "Normalised transport response"
        )
        bands = [
            collection
            for axis in figure.axes[4:]
            for collection in axis.collections
            if isinstance(collection, PolyCollection)
        ]
        assert len(bands) == 8
        assert [text.get_text() for text in figure.legends[0].get_texts()] == [
            "Semantic intervention",
            "Structural intervention",
        ]
    finally:
        plt.close(figure)


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
    loaded = cache.load(
        "diagnostic", {"task": "zinc", "head": [0, 1]}
    )
    assert loaded is not None
    assert loaded[0] == {"value": 1}
    assert loaded[1] == first_path
    assert cache.load("diagnostic", {"task": "missing"}) is None


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


def test_zinc_and_qm9_graphs_reconstruct_as_index_preserving_molecules():
    pytest.importorskip("rdkit")

    zinc = SimpleNamespace(
        x=np.asarray([[0], [0], [1]], dtype=np.int64),
        num_nodes=3,
        edge_index=np.asarray(
            [[0, 1, 1, 2], [1, 0, 2, 1]], dtype=np.int64
        ),
        edge_attr=np.asarray([1, 1, 1, 1], dtype=np.int64),
    )
    zinc_molecule = molecule_from_graph("zinc", zinc)
    zinc_payload = molecule_record("zinc", zinc)
    assert ZINC_ATOM_TYPES[4] == "C H1"
    assert graph_node_labels("zinc", zinc) == ["C", "C", "O"]
    assert [atom.GetSymbol() for atom in zinc_molecule.GetAtoms()] == [
        "C",
        "C",
        "O",
    ]
    assert zinc_payload["smiles"] == "CCO"
    assert zinc_payload["formula"] == "C2H6O"
    assert zinc_payload["chemistry_focus_version"] == CHEMISTRY_FOCUS_VERSION

    qm9 = SimpleNamespace(
        x=np.asarray([[8], [1], [1]], dtype=np.int64),
        num_nodes=3,
        edge_index=np.asarray(
            [[0, 1, 0, 2], [1, 0, 2, 0]], dtype=np.int64
        ),
        edge_attr=np.asarray([0, 0, 0, 0], dtype=np.int64),
        name="gdb_1",
    )
    qm9_payload = molecule_record("qm9_gap_dense", qm9)
    assert qm9_payload["formula"] == "H2O"
    assert qm9_payload["molecule_name"] == "gdb_1"
    assert figure_identity("qm9_gap_dense")["display_title"] == (
        "QM9 HOMO–LUMO gap — GRIT"
    )
    assert figure_identity("zinc") == {
        "dataset_label": "ZINC",
        "model_label": "GRIT",
        "display_title": "ZINC — GRIT",
    }


def test_pcqm_chemistry_categories_and_pca_colours_are_global():
    Chem = pytest.importorskip("rdkit.Chem")

    molecule = Chem.MolFromSmiles("CC(=O)O")
    categories = atom_chemistry_categories(molecule)
    assert categories[2] == "O: carbonyl"
    assert categories[3] == "O: ester/carboxyl"
    mass = np.asarray([0.0, 0.0, 0.9, 0.1])
    assert label_attention_focus(molecule, mass) == "O: carbonyl"
    assert _pca_focus_color("O: carbonyl") == PCA_FOCUS_COLORS["O: carbonyl"]
    assert _pca_focus_color("O: carbonyl") == "#C45100"
    assert _pca_focus_color("H: hydrogen") == "#D65F8D"
    assert (
        _pca_focus_color("H: hydrogen")
        != _pca_focus_color("other/diffuse")
    )
    assert _pca_focus_color("unknown category") == "#607080"


def test_grit_plotting_api_accepts_synthetic_payloads():
    import matplotlib.pyplot as plt
    Chem = pytest.importorskip("rdkit.Chem")

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
        # Simulate an existing supplemental cache from before the title
        # simplification; render-time task identity must take precedence.
        "dataset_label": "ZINC-subset",
        "model_label": "dense GRIT+RRWP",
        "display_title": "ZINC-subset — dense GRIT+RRWP",
        "examples": [
            {
                "dataset_index": 0,
                "n_atoms": 3,
                "mol_block": Chem.MolToMolBlock(Chem.MolFromSmiles("CON")),
                "smiles": "CON",
                "formula": "CH5NO",
                "molecule_name": None,
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
                "display_title": "ZINC — GRIT",
                "head": (0, 0),
                "vectors": np.arange(24, dtype=float).reshape(8, 3),
                "labels": ["Ring: aromatic"] * 4 + ["O: carbonyl"] * 4,
                "n_used": 8,
            }
        )
    )
    figures.append(
        plot_layer_av_pca_grid(
            {
                "task": "zinc",
                "display_title": "ZINC — GRIT",
                "layer": 0,
                "vectors": np.arange(96, dtype=float).reshape(8, 4, 3),
                "labels": [
                    ["Ring: aromatic", "O: carbonyl"] * 2
                    for _ in range(8)
                ],
                "n_used": 8,
            }
        )
    )
    logit = {
        "task": "zinc",
        "display_title": "ZINC-subset — dense GRIT+RRWP",
        "n_used": 5,
        "node_std_mean": np.asarray([0.5, 0.8]),
        "node_std_std": np.asarray([0.1, 0.1]),
        "relation_std_mean": np.asarray([0.7, 0.9]),
        "relation_std_std": np.asarray([0.1, 0.2]),
        "log_r_mean": np.zeros((2, 4)),
        "attention_entropy_mean": np.asarray(
            [[0.20, 0.35, 0.50, 0.65], [0.30, 0.45, 0.60, 0.75]]
        ),
        "attention_entropy_definition": (
            "mean query entropy normalised by log(supported key count)"
        ),
    }
    figures.append(plot_logit_spread(logit))
    figures.append(
        plot_selectivity_vs_logit_ratio(metrics, logit, selected)
    )
    figures.append(
        plot_selectivity_vs_attention_entropy(metrics, logit, selected)
    )
    figures.append(
        plot_joint_sensitivity_vs_attention_entropy(
            metrics, logit, selected
        )
    )
    np.testing.assert_allclose(
        figures[2].get_size_inches(),
        figures[3].get_size_inches(),
    )
    assert figures[4].axes[-1].get_xlabel() == "Attention weight"
    assert figures[4].axes[0].texts[0].get_text().startswith(
        "ZINC — GRIT — Semantic specialist"
    )
    assert figures[4].axes[1].texts[0].get_text().startswith("ZINC eval 0")
    np.testing.assert_allclose(figures[4].get_size_inches(), (13.2, 5.72))
    assert [text.get_fontsize() for text in figures[4].axes[0].texts] == [
        20,
        16,
    ]
    assert figures[4].axes[1].texts[0].get_fontsize() == 13
    assert figures[4].axes[3].xaxis.label.get_fontsize() == 13
    assert figures[4].axes[3].yaxis.label.get_fontsize() == 13
    assert all(
        label.get_fontsize() == 8
        for label in figures[4].axes[3].get_xticklabels()
    )
    assert figures[4].axes[-1].xaxis.label.get_fontsize() == 14
    assert all(
        label.get_fontsize() == 11
        for label in figures[4].axes[-1].get_xticklabels()
    )
    assert all(
        text.get_fontsize() == 9.5
        for text in figures[5].axes[0].get_legend().get_texts()
    )
    assert figures[6].legends[0].get_title().get_text() == "Attention focus"
    assert figures[7].axes[0].get_title().startswith("ZINC — GRIT\n")
    assert figures[8].axes[0].get_title().startswith("ZINC — GRIT\n")
    assert figures[9].axes[0].get_xlabel() == (
        "Mean normalised clean-attention entropy"
    )
    assert figures[9].axes[0].get_ylabel() == (
        r"Relative selectivity $D_{\rm rel}$"
    )
    assert figures[10].axes[0].get_ylabel() == r"Joint sensitivity $J$"
    assert all(figure.axes for figure in figures)
    for figure in figures:
        plt.close(figure)


def test_pca_legend_is_ranked_by_observed_frequency():
    import matplotlib.pyplot as plt

    labels = (
        ["Ring: aromatic"] * 2
        + ["O: carbonyl"] * 5
        + ["N: amide"] * 3
        + ["other/diffuse"] * 6
    )
    figure = plot_av_pca(
        {
            "task": "zinc",
            "display_title": "ZINC — GRIT",
            "head": (0, 0),
            "vectors": np.arange(len(labels) * 3, dtype=float).reshape(
                len(labels), 3
            ),
            "labels": labels,
            "n_used": len(labels),
        }
    )
    try:
        observed = [
            text.get_text().split(" (n=", 1)[0]
            for text in figure.axes[0].get_legend().get_texts()
        ]
        assert observed == [
            "O: carbonyl",
            "N: amide",
            "Ring: aromatic",
            "other/diffuse",
        ]
    finally:
        plt.close(figure)


def test_pcqm_aligned_figure_dimensions():
    import matplotlib.pyplot as plt

    metrics = CanonicalHeadMetrics.from_scores(_score_value())
    distance_metrics = dataclasses.replace(
        metrics,
        distance_axis=("0", "1", "2"),
        clean_attention_distance=np.full((2, 4, 3), 1 / 3),
    )
    selected = {"semantic": (0, 0), "structural": (0, 1)}
    logit = {
        "task": "zinc",
        "display_title": "ZINC — GRIT",
        "n_used": 5,
        "node_std_mean": np.asarray([0.5, 0.8]),
        "node_std_std": np.asarray([0.1, 0.1]),
        "relation_std_mean": np.asarray([0.7, 0.9]),
        "relation_std_std": np.asarray([0.1, 0.2]),
        "log_r_mean": np.zeros((2, 4)),
        "attention_entropy_mean": np.asarray(
            [[0.20, 0.35, 0.50, 0.65], [0.30, 0.45, 0.60, 0.75]]
        ),
        "attention_entropy_definition": (
            "mean query entropy normalised by log(supported key count)"
        ),
    }
    figures_and_sizes = [
        (
            plot_score_heatmaps(
                metrics, selected, title="ZINC — GRIT"
            ),
            (14.0, 5.2),
        ),
        (
            plot_coordinate_heatmaps(
                metrics, selected, title="ZINC — GRIT"
            ),
            (14.0, 5.2),
        ),
        (
            plot_score_plane(
                metrics, selected, title="ZINC — GRIT"
            ),
            (8.0, 5.9),
        ),
        (
            plot_selectivity_joint_plane(
                metrics, selected, title="ZINC — GRIT"
            ),
            (8.0, 5.9),
        ),
        (
            plot_hop_attention_mass(distance_metrics, (0, 0)),
            (8.4, 4.8),
        ),
        (plot_logit_spread(logit), (8.2, 4.9)),
        (
            plot_selectivity_vs_logit_ratio(metrics, logit, selected),
            (8.1, 5.8),
        ),
        (
            plot_selectivity_vs_attention_entropy(
                metrics, logit, selected
            ),
            (8.1, 5.8),
        ),
        (
            plot_joint_sensitivity_vs_attention_entropy(
                metrics, logit, selected
            ),
            (8.1, 5.8),
        ),
    ]
    try:
        for figure, expected_size in figures_and_sizes:
            np.testing.assert_allclose(
                figure.get_size_inches(),
                expected_size,
            )
    finally:
        for figure, _ in figures_and_sizes:
            plt.close(figure)


def test_coordinate_heatmap_header_is_visible_and_separate_from_panel_titles():
    import matplotlib.pyplot as plt

    figure = plot_coordinate_heatmaps(
        CanonicalHeadMetrics.from_scores(_score_value()),
        title="ZINC — GRIT",
    )
    try:
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        header_box = figure._suptitle.get_window_extent(renderer)
        panel_title_boxes = [
            axis.title.get_window_extent(renderer) for axis in figure.axes[:2]
        ]
        assert header_box.y1 <= figure.bbox.y1
        assert header_box.y0 > max(box.y1 for box in panel_title_boxes)
        assert figure._suptitle.get_fontsize() == 18
        assert all(axis.title.get_fontsize() == 13 for axis in figure.axes[:2])
        assert [axis.yaxis.label.get_fontsize() for axis in figure.axes[2:]] == [
            11,
            11,
        ]
        assert all(
            label.get_fontsize() == 9
            for axis in figure.axes[2:]
            for label in axis.get_yticklabels()
        )
    finally:
        plt.close(figure)


def test_section_pdf_bundles_are_task_prefixed_and_ordered(tmp_path: Path):
    import matplotlib.pyplot as plt

    PdfReader = pytest.importorskip("pypdf").PdfReader
    pages = []
    for index in range(2):
        figure, axis = plt.subplots()
        axis.text(0.5, 0.5, f"page {index}", ha="center")
        paths = save_figure_bundle(
            figure,
            tmp_path / "individual",
            f"figure_{index}",
            dpi=72,
            pdf_dpi=72,
        )
        pages.append((f"figure_{index}", paths["pdf"]))
        plt.close(figure)
    outputs = save_section_pdf_bundles(
        {"semantic_specialists": pages},
        tmp_path / "sections",
        task_prefix="zinc",
        section_titles={"semantic_specialists": "Semantic specialists"},
    )
    record = outputs["semantic_specialists"]
    assert record["path"].name == "zinc_semantic_specialists.pdf"
    assert record["pages"] == 2
    assert record["figure_stems"] == ["figure_0", "figure_1"]
    assert len(PdfReader(str(record["path"])).pages) == 2


def test_figure_bundle_uses_publication_export_resolution(tmp_path: Path):
    class RecordingFigure:
        def __init__(self):
            self.calls = []

        def savefig(self, path, **kwargs):
            self.calls.append((Path(path), kwargs))
            Path(path).write_bytes(b"test")

    figure = RecordingFigure()
    save_figure_bundle(figure, tmp_path, "publication")
    assert figure.calls[0][0].suffix == ".png"
    assert figure.calls[0][1]["dpi"] == PUBLICATION_PNG_DPI == 600
    assert figure.calls[1][0].suffix == ".pdf"
    assert figure.calls[1][1]["dpi"] == PUBLICATION_PDF_RASTER_DPI == 1200
    assert MOLECULE_DRAW_DPI == MOLECULE_RENDER_DPI == 600
    assert figure.calls[1][1]["metadata"]["Title"] == "publication"
    metadata = json.loads((tmp_path / "publication.json").read_text())
    assert metadata["export_quality"] == {
        "png_dpi": 600,
        "pdf_raster_dpi": 1200,
        "pdf_vector_text_and_paths": True,
        "pdf_font_embedding": "TrueType (fonttype 42)",
        "molecule_render_dpi": 600,
    }


def test_core_scatter_exports_keep_matching_tight_geometry(tmp_path: Path):
    import matplotlib.pyplot as plt

    metrics = CanonicalHeadMetrics.from_scores(_score_value())
    selected = {"semantic": (0, 0), "structural": (0, 1)}
    figures = [
        plot_score_plane(metrics, selected, title="Synthetic GRIT"),
        plot_selectivity_joint_plane(
            metrics, selected, title="Synthetic GRIT"
        ),
    ]
    exported_shapes = []
    try:
        assert figures[0].axes[0].get_aspect() == "auto"
        for index, figure in enumerate(figures):
            paths = save_figure_bundle(
                figure,
                tmp_path,
                f"scatter_{index}",
                dpi=72,
                pdf_dpi=72,
            )
            exported_shapes.append(plt.imread(paths["png"]).shape[:2])
        assert exported_shapes[0] == exported_shapes[1]
    finally:
        for figure in figures:
            plt.close(figure)


def test_layer_pca_collection_reuses_one_grit_forward_per_graph(monkeypatch):
    torch = pytest.importorskip("torch")
    num_layers = 3
    num_heads = 4
    num_nodes = 3
    head_width = 2
    graphs = [
        SimpleNamespace(num_nodes=num_nodes, graph_index=index)
        for index in range(3)
    ]
    runtime = SimpleNamespace(
        L=num_layers,
        H=num_heads,
        eval_ds=graphs,
    )
    figure_runtime = SimpleNamespace(
        runtime=runtime,
        prepared=SimpleNamespace(task=SimpleNamespace(name="zinc")),
    )
    captured = SimpleNamespace(
        transport=tuple(
            torch.arange(
                num_nodes * num_heads * head_width,
                dtype=torch.float32,
            ).reshape(num_nodes, num_heads, head_width)
            + layer
            for layer in range(num_layers)
        ),
        attention=tuple(
            torch.ones(
                num_heads,
                num_nodes,
                num_nodes,
                dtype=torch.float32,
            )
            for _ in range(num_layers)
        ),
    )
    calls = []

    class FakeExtractor:
        def __init__(self, received):
            assert received is figure_runtime

        def extract(self, graph):
            calls.append(graph.graph_index)
            return captured

    monkeypatch.setattr(
        "graph_specialisation_metrics.methodology.grit_figure_data."
        "GritDiagnosticExtractor",
        FakeExtractor,
    )
    molecule = object()
    monkeypatch.setattr(
        "graph_specialisation_metrics.methodology.grit_figure_data."
        "molecule_from_graph",
        lambda task_name, graph: molecule,
    )
    monkeypatch.setattr(
        "graph_specialisation_metrics.methodology.grit_figure_data."
        "label_attention_focus",
        lambda received, mass, **kwargs: "Ring: aromatic",
    )
    payload = compute_layer_av_pca_inputs(
        figure_runtime,
        layers=(0, 2),
        n_graphs=3,
        verbose=False,
    )

    assert calls == [0, 1, 2]
    assert payload["requested_layers"] == (0, 2)
    assert payload["num_heads"] == num_heads
    assert payload["n_used"] == 3
    for layer in (0, 2):
        layer_payload = payload["layers"][layer]
        assert layer_payload["vectors"].shape == (
            3,
            num_heads,
            head_width,
        )
        assert np.asarray(layer_payload["labels"]).shape == (3, num_heads)
        assert {
            label
            for row in layer_payload["labels"]
            for label in row
        } == {"Ring: aromatic"}


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
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in payload["cells"]
    )
    assert '"rdkit"' in source
    assert '"pypdf"' in source
    assert 'TASK_SELECTION = "zinc"  # @param ["zinc", "qm9", "both"]' in source
    assert '"zinc": ("zinc",)' in source
    assert '"qm9": ("qm9_gap_dense",)' in source
    assert "HEADS_PER_FAMILY = 5" in source
    assert 'f"semantic_{rank}": "Semantic specialist"' in source
    assert 'f"structural_{rank}": "Structural specialist"' in source
    assert 'f"generalist_{rank}": r"High-$J$ generalist"' in source
    assert "specialist {rank}" not in source
    assert "generalist {rank}" not in source
    assert "PNG_DPI = 600" in source
    assert "PDF_RASTER_DPI = 1200" in source
    assert "del sys.modules[module_name]" in source
    assert "def get_figure_runtime()" in source
    assert "if figure_runtime is None" in source
    assert "structural_count=HEADS_PER_FAMILY + 1" in source
    assert 'all_roles = dict(context["display_heads"])' in source
    assert "clean_attention_mass_vs_SPD" in source
    runtime_source = "".join(payload["cells"][6]["source"])
    assert runtime_source.index("plot_attention_grid(") < runtime_source.index(
        "plot_av_pca("
    )
    assert runtime_source.index("plot_av_pca(") < runtime_source.index(
        'f"{role}_head_{head[0]}_{head[1]}_clean_attention_mass_vs_SPD"'
    )
    assert "companion_heads = set(all_roles.values())" not in runtime_source
    assert "additional structural head" not in runtime_source
    assert "compute_layer_av_pca_inputs(" in runtime_source
    assert "plot_layer_av_pca_grid(" in runtime_source
    assert "plot_selectivity_vs_attention_entropy(" in runtime_source
    assert "plot_joint_sensitivity_vs_attention_entropy(" in runtime_source
    assert "GRIT_raw_attention_logit_spread_and_entropy_v2" in runtime_source
    assert (
        "grit-logit-spread-and-attention-entropy-v2" in runtime_source
    )
    assert "BOOTSTRAP_REPLICATES" in source
    assert "ZINC_TRANSPORT_PROFILE_HEAD_GROUPS" in source
    assert "((1, 2), (1, 7), (4, 7), (6, 0))" in source
    assert "((6, 3), (7, 6), (9, 1), (8, 4))" in source
    assert "compute_selected_head_transport_profiles(" in runtime_source
    assert "plot_routing_transport_profiles(" in runtime_source
    assert 'if task_name == "zinc" else ()' in runtime_source
    assert "selected-head-transport-response-by-distance" in runtime_source
    assert (
        'f"routing_geometry_vs_transport_response_zinc_heads_{group_index}"'
        in runtime_source
    )
    assert 'PDF_TASK_PREFIXES = {"zinc": "zinc", "qm9_gap_dense": "qm9"}' in source
    assert "save_section_pdf_bundles(" in runtime_source
    assert '"semantic_specialists"' in source
    assert '"distance_curves"' in source
    assert '"layer_pca_overviews"' in source
    assert 'RUN_ROOT / "pdf_sections"' in source
    for index, cell in enumerate(payload["cells"]):
        if cell["cell_type"] == "code":
            ast.parse(
                "".join(cell["source"]),
                filename=f"{notebook}:cell_{index}",
            )
