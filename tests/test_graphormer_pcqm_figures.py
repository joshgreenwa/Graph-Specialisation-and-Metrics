from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
from matplotlib.collections import PathCollection, PolyCollection
from matplotlib.colors import to_rgba
from matplotlib.container import ErrorbarContainer
from matplotlib.patches import FancyArrowPatch
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from graph_specialisation_metrics.methodology import runner
from graph_specialisation_metrics.methodology.cache import (
    CacheCompatibilityWarning,
    StaleCacheError,
    load_cache_artifact_file,
)
from graph_specialisation_metrics.methodology.graphormer import (
    GraphormerBackend,
    GraphormerRuntime,
    graphormer_graph_from_ogb,
)
from graph_specialisation_metrics.methodology.graphormer_figure_data import (
    CanonicalHeadMetrics,
    GraphormerDiagnosticExtractor,
    SupplementalCache,
    compute_layer_av_pca_inputs,
    load_graphormer_model_record,
    load_graphormer_score_artifact,
    select_attention_examples_for_grid,
    select_attention_grid_indices,
    select_ranked_heads,
    select_specialist_heads,
    select_structural_specialist_head,
)
from graph_specialisation_metrics.methodology.graphormer_figure_plots import (
    MOLECULE_RENDER_DPI,
    PCA_FOCUS_COLORS,
    PUBLICATION_PDF_RASTER_DPI,
    PUBLICATION_PNG_DPI,
    plot_attention_grid,
    plot_av_pca,
    plot_coordinate_heatmaps,
    plot_hop_attention_mass,
    plot_layer_av_pca_grid,
    plot_logit_spread,
    plot_score_heatmaps,
    plot_score_plane,
    plot_selectivity_joint_plane,
    plot_selectivity_vs_logit_ratio,
    save_figure_bundle,
)
from graph_specialisation_metrics.methodology.tasks import get_task
from graph_specialisation_metrics.methodology.protocol import (
    ExecutionPolicy,
    MethodologyConfig,
    PROTOCOL_VERSION,
    SplitManifest,
    stable_hash,
)


def synthetic_metrics() -> CanonicalHeadMetrics:
    semantic = np.asarray([[0.7, 1.1, 1.3], [0.8, 1.0, 2.0]])
    structural = np.asarray([[1.8, 1.0, 0.9], [1.4, 0.8, 0.7]])
    joint = 0.5 * (semantic + structural)
    selectivity = (semantic - structural) / (semantic + structural)
    coordinates = SimpleNamespace(
        raw_semantic=semantic * 0.2,
        raw_structural=structural * 0.3,
        normalized_semantic=semantic,
        normalized_structural=structural,
        joint_sensitivity=joint,
        selectivity=selectivity,
        active=np.asarray([[True, True, True], [True, False, True]]),
        estimable=True,
    )
    return CanonicalHeadMetrics.from_scores(
        {
            "coordinates": coordinates,
            "channels": {
                "semantic": {"raw": coordinates.raw_semantic.copy()},
                "structural": {"raw": coordinates.raw_structural.copy()},
            },
            "axis": (0, 1, 2, "graph_token"),
            "clean_attention_distance": np.full((2, 3, 4), 0.25),
        }
    )


def tiny_graph(config):
    graph = {
        "num_nodes": 3,
        "node_feat": np.asarray(
            [[5, 0, 4], [6, 1, 3], [7, 2, 2]], dtype=np.int64
        ),
        "edge_index": np.asarray(
            [[0, 1, 1, 2], [1, 0, 2, 1]], dtype=np.int64
        ),
        "edge_feat": np.asarray(
            [[0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]],
            dtype=np.int64,
        ),
    }
    return graphormer_graph_from_ogb(
        graph, [0.25], config=config, smiles="CCO"
    )


def test_canonical_adapter_and_active_drel_selection():
    metrics = synthetic_metrics()
    selected = select_specialist_heads(metrics, semantic_head=(1, 2))
    assert metrics.shape == (2, 3)
    assert selected == {"semantic": (1, 2), "structural": (0, 0)}
    assert metrics.distance_axis[-1] == "graph_token"
    assert get_task("graphormer_pcqm4mv2").title == "Graphormer PCQM4Mv2"


def test_structural_selection_can_exclude_an_entire_head_index():
    metrics = synthetic_metrics()
    primary = select_structural_specialist_head(metrics)
    alternative = select_structural_specialist_head(
        metrics,
        excluded_heads=(primary,),
        excluded_head_indices=(primary[1],),
    )
    assert primary == (0, 0)
    assert alternative == (0, 1)
    assert alternative[1] != primary[1]


def test_attention_grid_indices_require_enough_rows_and_truncate_extras():
    assert select_attention_grid_indices(
        [11, 22, 33, 44, 55],
        num_rows=4,
    ) == [11, 22, 33, 44]
    assert select_attention_grid_indices(
        [0, 5, 80, 100, 200, 300],
        num_rows=5,
    ) == [0, 5, 80, 100, 200]
    with pytest.raises(ValueError, match="needs 5 graph indices"):
        select_attention_grid_indices([0, 5, 80], num_rows=5)
    with pytest.raises(ValueError, match="must be positive"):
        select_attention_grid_indices([0], num_rows=0)
    with pytest.raises(ValueError, match="must be unique"):
        select_attention_grid_indices([0, 5, 5, 80], num_rows=4)


def test_attention_grid_payload_is_reordered_and_trimmed_to_configuration():
    payload = {
        "heads": {"semantic": (1, 24)},
        "examples": [
            {"dataset_index": index, "value": f"graph-{index}"}
            for index in (200, 80, 0, 100, 5)
        ],
    }
    selected = select_attention_examples_for_grid(
        payload,
        graph_indices=[5, 100, 0, 80],
    )

    assert payload["examples"][0]["dataset_index"] == 200
    assert selected["heads"] == payload["heads"]
    assert [example["dataset_index"] for example in selected["examples"]] == [
        5,
        100,
        0,
        80,
    ]
    with pytest.raises(ValueError, match="does not contain"):
        select_attention_examples_for_grid(
            payload,
            graph_indices=[5, 999],
        )


def test_graphormer_figure_notebook_routes_every_grid_through_live_config():
    notebook_path = (
        Path(__file__).parents[1]
        / "experiments"
        / "methodology"
        / "graphormer_pcqm4mv2_figures_colab.ipynb"
    )
    notebook = json.loads(notebook_path.read_text())
    source = "\n".join(
        "".join(cell.get("source", ()))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )

    assert "ATTENTION_GRID_NUM_ROWS = 4" in source
    assert "PNG_DPI = 600" in source
    assert "PDF_RASTER_DPI = 1200" in source
    assert "pdf_dpi=PDF_RASTER_DPI" in source
    assert (
        "ATTENTION_GRID_GRAPH_INDICES[role],\n"
        "        num_rows=ATTENTION_GRID_NUM_ROWS"
    ) in source
    assert "select_attention_examples_for_grid(" in source
    assert "rendering {len(graph_indices)} rows" in source
    assert "del sys.modules[module_name]" in source
    assert (
        'attention_stem = f"{role}_head_{head[0]}_{head[1]}_attention_grid"'
        in source
    )
    assert "attention_{ATTENTION_GRID_NUM_ROWS}x3" not in source


def test_graphormer_figure_loader_explicitly_accepts_valid_v3_cache(tmp_path):
    contract = {
        "task": "graphormer_pcqm4mv2",
        "task_adapter_version": "canonical-graphormer-hf-v1",
        "checkpoint_sha256": "checkpoint",
        "train_seed": 0,
        "model_geometry": {"layers": 12, "heads": 32},
        "sigma": [1.0],
        "split_fingerprint": "split",
    }
    path = tmp_path / "raw.pt"
    torch.save(
        {
            "metadata": {
                "protocol_version": "donor-swap-specialisation-carriage-v3",
                "contract": contract,
                "contract_fingerprint": stable_hash(contract),
            },
            "value": {"coordinates": "test"},
        },
        path,
    )

    with pytest.warns(CacheCompatibilityWarning, match="without relabelling"):
        artifact = load_cache_artifact_file(path)
    assert artifact.metadata["protocol_version"].endswith("-v3")
    with pytest.raises(StaleCacheError, match="without relabelling"):
        load_cache_artifact_file(path, strict_protocol=True)
    with pytest.warns(CacheCompatibilityWarning, match="without relabelling"):
        artifact = load_graphormer_score_artifact(path)
    assert artifact.metadata["protocol_version"].endswith("-v3")
    assert artifact.metadata["contract"]["task"] == "graphormer_pcqm4mv2"

    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["metadata"]["contract_fingerprint"] = "corrupt"
    torch.save(payload, path)
    with pytest.raises(StaleCacheError, match="internally inconsistent"):
        load_graphormer_score_artifact(path)


def test_model_record_is_bound_to_score_cache_split(tmp_path):
    splits = SplitManifest(
        discovery=(0, 1),
        causal=(2,),
        clean_ablation=(3,),
        semantic_donor_pool=(4, 5, 6),
        same_index_space=False,
        seed=31_415,
    )
    contract = {
        "task": "graphormer_pcqm4mv2",
        "task_adapter_version": "canonical-graphormer-hf-v1",
        "checkpoint_sha256": "checkpoint",
        "train_seed": 0,
        "model_geometry": {"layers": 12, "heads": 32},
        "sigma": [1.0],
        "split_fingerprint": splits.fingerprint,
        "source_cap": 6,
        "donors_per_source": 8,
    }
    score_path = tmp_path / "raw.pt"
    torch.save(
        {
            "metadata": {
                "protocol_version": PROTOCOL_VERSION,
                "contract": contract,
                "contract_fingerprint": stable_hash(contract),
            },
            "value": {"coordinates": "test"},
        },
        score_path,
    )
    artifact = load_graphormer_score_artifact(score_path)
    model_path = tmp_path / "model.json"
    model_record = {
        "task": contract["task"],
        "task_adapter_version": contract["task_adapter_version"],
        "checkpoint_sha256": contract["checkpoint_sha256"],
        "train_seed": contract["train_seed"],
        "splits": {
            "discovery": list(splits.discovery),
            "causal": list(splits.causal),
            "clean_ablation": list(splits.clean_ablation),
            "semantic_donor_pool": list(splits.semantic_donor_pool),
            "same_index_space": splits.same_index_space,
            "seed": splits.seed,
        },
    }
    model_path.write_text(json.dumps(model_record))
    loaded = load_graphormer_model_record(model_path, artifact)
    assert loaded["splits"]["seed"] == 31_415

    model_record["splits"]["semantic_donor_pool"] = [4, 5]
    model_path.write_text(json.dumps(model_record))
    with pytest.raises(StaleCacheError, match="split fingerprint"):
        load_graphormer_model_record(model_path, artifact)


def test_ranked_head_selection_is_independent_and_deterministic():
    ranked = select_ranked_heads(synthetic_metrics())
    assert ranked == {
        "top_semantic_1": (1, 2),
        "top_semantic_2": (0, 2),
        "top_joint_1": (1, 2),
        "top_joint_2": (0, 0),
    }


def test_ranked_selection_adds_third_semantic_and_high_j_generalists():
    ranked = select_ranked_heads(
        synthetic_metrics(),
        semantic_count=3,
        joint_count=3,
        joint_generalist_max_abs_selectivity=0.3,
    )
    assert ranked == {
        "top_semantic_1": (1, 2),
        "top_semantic_2": (0, 2),
        "top_semantic_3": (0, 1),
        "top_joint_1": (0, 2),
        "top_joint_2": (1, 0),
        "top_joint_3": (0, 1),
    }
    metrics = synthetic_metrics()
    assert all(
        abs(metrics.selectivity[head]) <= 0.3
        for role, head in ranked.items()
        if role.startswith("top_joint")
    )


def test_supplemental_cache_is_immutable_and_contract_keyed(tmp_path):
    calls = []
    cache = SupplementalCache(tmp_path)
    first, first_path, first_hit = cache.load_or_compute(
        "demo", {"n": 3}, lambda: calls.append(1) or {"value": 7}
    )
    second, second_path, second_hit = cache.load_or_compute(
        "demo", {"n": 3}, lambda: calls.append(2) or {"value": 9}
    )
    third, third_path, third_hit = cache.load_or_compute(
        "demo", {"n": 4}, lambda: calls.append(3) or {"value": 11}
    )
    assert first == second
    assert first_path == second_path
    assert first_path != third_path
    assert not first_hit
    assert second_hit
    assert not third_hit
    assert calls == [1, 3]
    assert third["value"] == 11


def test_graph_local_estimator_normalises_each_graph_independently(monkeypatch):
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
        *,
        include_diagnostics,
    ):
        assert axis is None
        assert include_diagnostics is False
        values = semantic if channel == "semantic" else structural
        return [
            {"graph_id": int(graph_id), "score": values[int(graph_id)]}
            for graph_id in graph_ids
        ]

    monkeypatch.setattr(runner, "_score_graph_batch", fake_score_batch)
    monkeypatch.setattr(runner, "_event_item_cost", lambda *args, **kwargs: 1)
    backend = SimpleNamespace(
        clean_jacobians_many=lambda bases: [f"clean-{base}" for base in bases]
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

    first = result["graphs"][10]
    second = result["graphs"][20]
    assert np.isclose(first["normalized_semantic"].mean(), 1.0)
    assert np.isclose(first["normalized_structural"].mean(), 1.0)
    assert np.isclose(second["normalized_semantic"].mean(), 1.0)
    assert np.isclose(second["normalized_structural"].mean(), 1.0)
    assert not np.allclose(first["selectivity"], second["selectivity"])
    assert result["graph_id_seed_space"] == "caller-supplied graph ID"


def test_requested_scatter_figures_have_matching_export_geometry(tmp_path):
    metrics = synthetic_metrics()
    selected = {"semantic": (1, 2), "structural": (0, 0)}
    figures = (
        plot_score_plane(metrics, selected),
        plot_selectivity_joint_plane(
            metrics, selected, xlim=(-0.6, 0.6)
        ),
    )
    try:
        assert np.allclose(figures[0].get_size_inches(), (8.0, 5.9))
        assert np.allclose(
            figures[0].get_size_inches(),
            figures[1].get_size_inches(),
        )
        assert figures[0].axes[0].get_aspect() == "auto"
        exported_shapes = []
        for index, figure in enumerate(figures):
            paths = save_figure_bundle(
                figure,
                tmp_path,
                f"scatter_{index}",
                dpi=72,
                pdf_dpi=72,
            )
            exported_shapes.append(plt.imread(paths["png"]).shape[:2])
            containers = [
                container
                for axis in figure.axes
                for container in axis.containers
                if isinstance(container, ErrorbarContainer)
            ]
            assert containers == []
            highlight_rings = [
                collection
                for axis in figure.axes
                for collection in axis.collections
                if isinstance(collection, PathCollection)
                and np.any(np.isclose(collection.get_sizes(), 110))
            ]
            assert len(highlight_rings) == 2
            assert all(len(ring.get_facecolors()) == 0 for ring in highlight_rings)
            specialist_annotations = {
                text.get_text().split()[0].lower(): text
                for axis in figure.axes
                for text in axis.texts
                if "specialist" in text.get_text()
            }
            assert set(specialist_annotations) == {"semantic", "structural"}
            for label, expected_color in (
                ("semantic", "#E6A700"),
                ("structural", "#087E8B"),
            ):
                patch = specialist_annotations[label].get_bbox_patch()
                assert patch is not None
                assert np.allclose(
                    patch.get_facecolor(),
                    to_rgba("white", alpha=0.94),
                )
                assert np.allclose(
                    patch.get_edgecolor()[:3],
                    to_rgba(expected_color)[:3],
                )
        assert exported_shapes[0] == exported_shapes[1]
    finally:
        for figure in figures:
            plt.close(figure)


def test_coordinate_heatmaps_mask_inactive_selectivity():
    metrics = synthetic_metrics()
    figure = plot_coordinate_heatmaps(metrics)
    try:
        assert figure._suptitle.get_text() == "Graphormer PCQM4Mv2"
        image_axes = [axis for axis in figure.axes if axis.images]
        assert len(image_axes) == 2
        rendered = image_axes[0].images[0].get_array()
        assert bool(np.ma.getmaskarray(rendered)[1, 1])
        assert image_axes[0].images[0].get_cmap().name == "coolwarm"
        assert image_axes[1].images[0].get_cmap().name == "viridis"
    finally:
        plt.close(figure)


def test_score_heatmaps_and_scatter_restore_viridis():
    metrics = synthetic_metrics()
    heatmaps = plot_score_heatmaps(metrics)
    scatter = plot_score_plane(metrics)
    try:
        assert plt.rcParams["savefig.dpi"] == PUBLICATION_PNG_DPI
        assert plt.rcParams["pdf.fonttype"] == 42
        assert plt.rcParams["pdf.compression"] == 9
        assert heatmaps._suptitle.get_text() == "Graphormer PCQM4Mv2"
        assert heatmaps.axes[-1].get_ylabel() == "Normalised score"
        assert scatter.axes[0].get_title() == "Graphormer PCQM4Mv2"
        assert all(
            axis.images[0].get_cmap().name == "viridis"
            for axis in heatmaps.axes
            if axis.images
        )
        layer_collection = next(
            collection
            for collection in scatter.axes[0].collections
            if collection.get_array() is not None
        )
        assert layer_collection.get_cmap().name == "viridis"
    finally:
        plt.close(heatmaps)
        plt.close(scatter)


def test_heatmaps_do_not_outline_selected_heads():
    metrics = synthetic_metrics()
    selected = {"semantic": (1, 2), "structural": (0, 0)}
    figures = (
        plot_score_heatmaps(metrics, selected),
        plot_coordinate_heatmaps(metrics, selected),
    )
    try:
        for figure in figures:
            image_axes = [axis for axis in figure.axes if axis.images]
            assert len(image_axes) == 2
            assert all(len(axis.patches) == 0 for axis in image_axes)
    finally:
        for figure in figures:
            plt.close(figure)


def test_figure_bundle_saves_png_pdf_and_provenance_in_target_folder(tmp_path):
    assert save_figure_bundle.__kwdefaults__["dpi"] == PUBLICATION_PNG_DPI
    assert (
        save_figure_bundle.__kwdefaults__["pdf_dpi"]
        == PUBLICATION_PDF_RASTER_DPI
    )
    assert MOLECULE_RENDER_DPI == 600
    figure = plot_score_plane(synthetic_metrics())
    target = tmp_path / "semantic_specialists"
    try:
        paths = save_figure_bundle(
            figure,
            target,
            "example_head",
            metadata={"figure_group": "semantic_specialists"},
            dpi=100,
            pdf_dpi=200,
        )
    finally:
        plt.close(figure)

    assert paths == {
        "png": target / "example_head.png",
        "pdf": target / "example_head.pdf",
        "metadata": target / "example_head.json",
    }
    assert all(path.is_file() for path in paths.values())
    metadata = json.loads(paths["metadata"].read_text())
    assert metadata["figure_group"] == "semantic_specialists"
    assert metadata["export_quality"] == {
        "png_dpi": 100,
        "pdf_raster_dpi": 200,
        "pdf_vector_text_and_paths": True,
        "pdf_font_embedding": "TrueType (fonttype 42)",
        "molecule_render_dpi": 600,
    }


def test_figure_bundle_uses_independent_publication_dpi_for_png_and_pdf(
    tmp_path,
):
    calls = []

    class RecordingFigure:
        def savefig(self, path, **kwargs):
            calls.append((Path(path), kwargs))
            Path(path).write_bytes(b"figure")

    paths = save_figure_bundle(
        RecordingFigure(),
        tmp_path,
        "publication_export",
        dpi=600,
        pdf_dpi=1200,
    )

    assert calls[0][0] == paths["png"]
    assert calls[0][1]["dpi"] == 600
    assert calls[1][0] == paths["pdf"]
    assert calls[1][1]["dpi"] == 1200
    assert calls[1][1]["metadata"] == {
        "Title": "publication_export",
        "Creator": "Graph Specialisation and Metrics",
        "Subject": "Publication figure",
    }


def test_figure_bundle_removes_only_superseded_generated_outputs(tmp_path):
    target = tmp_path / "semantic_specialists"
    target.mkdir()
    legacy_stem = "semantic_head_1_24_attention_5x3"
    for suffix in (".png", ".pdf", ".json"):
        (target / f"{legacy_stem}{suffix}").write_text("replaceable")
    unrelated = target / "semantic_head_1_24_attention_notes.txt"
    unrelated.write_text("keep")
    figure = plot_score_plane(synthetic_metrics())
    try:
        paths = save_figure_bundle(
            figure,
            target,
            "semantic_head_1_24_attention_grid",
            dpi=72,
            pdf_dpi=72,
            supersede_stem_globs=(
                "semantic_head_1_24_attention_*x3.*",
            ),
        )
    finally:
        plt.close(figure)

    assert all(path.is_file() for path in paths.values())
    assert not any(
        (target / f"{legacy_stem}{suffix}").exists()
        for suffix in (".png", ".pdf", ".json")
    )
    assert unrelated.read_text() == "keep"


def test_hop_plot_separates_graph_token_tick():
    figure = plot_hop_attention_mass(synthetic_metrics(), (0, 0))
    try:
        axis = figure.axes[0]
        ticks = axis.get_xticks()
        labels = [label.get_text() for label in axis.get_xticklabels()]
        assert labels[-1] == "Graph\ntoken"
        assert ticks[-1] - ticks[-2] > 1.2
    finally:
        plt.close(figure)


def test_logit_plot_has_uncertainty_bands_and_ratio_is_inverted():
    metrics = synthetic_metrics()
    payload = {
        "dot_std_mean": np.asarray([0.7, 0.9]),
        "dot_std_std": np.asarray([0.1, 0.12]),
        "bias_std_mean": np.asarray([1.1, 1.0]),
        "bias_std_std": np.asarray([0.2, 0.15]),
        "log_r_mean": np.asarray(
            [[-0.4, -0.2, 0.0], [0.1, 0.3, 0.5]]
        ),
        "n_used": 100,
    }
    spread = plot_logit_spread(payload)
    ratio = plot_selectivity_vs_logit_ratio(metrics, payload)
    try:
        bands = [
            collection
            for collection in spread.axes[0].collections
            if isinstance(collection, PolyCollection)
        ]
        assert len(bands) == 2
        layer_collection = next(
            collection
            for collection in ratio.axes[0].collections
            if collection.get_array() is not None
        )
        actual_x = np.sort(np.asarray(layer_collection.get_offsets())[:, 0])
        expected_x = np.sort(-payload["log_r_mean"][metrics.active])
        assert np.allclose(actual_x, expected_x)
        assert r"\mathrm{std}(d)/\mathrm{std}(b)" in ratio.axes[0].get_xlabel()
        assert ratio.axes[0].get_ylabel() == (
            r"Relative selectivity $D_{\rm rel}$"
        )
    finally:
        plt.close(spread)
        plt.close(ratio)


def test_attention_grid_supports_five_rows_without_arrows():
    pytest.importorskip("rdkit")
    attention = np.asarray(
        [
            [0.25, 0.25, 0.25, 0.25],
            [0.05, 0.60, 0.25, 0.10],
            [0.05, 0.20, 0.55, 0.20],
            [0.05, 0.15, 0.25, 0.55],
        ]
    )
    payload = {
        "examples": [
            {
                "dataset_index": index,
                "smiles": "CCO",
                "attention": {"semantic": attention},
            }
            for index in (0, 5, 80, 100, 200)
        ]
    }
    figure = plot_attention_grid(
        payload,
        role="semantic",
        head=(1, 24),
        per_graph_coordinates={
            0: {"D_rel": 0.4815, "J": 1.35},
            5: {"D_rel": 0.102, "J": 0.93},
            80: {"D_rel": -0.221, "J": 1.71},
            100: {"D_rel": 0.088, "J": 1.11},
            200: {"D_rel": -0.041, "J": 0.84},
        },
        net_d_rel=0.4815,
        net_joint_sensitivity=1.35,
        title_label="Semantic specialist",
    )
    try:
        assert not any(
            isinstance(patch, FancyArrowPatch)
            for axis in figure.axes
            for patch in axis.patches
        )
        weighted_axis = next(
            axis
            for axis in figure.axes
            if axis.get_title() == "Attention-weighted molecule"
        )
        matrix_axis = next(
            axis
            for axis in figure.axes
            if axis.get_title() == "Node-conditioned attention"
        )
        assert weighted_axis.images
        assert matrix_axis.images[0].get_cmap().name == "Blues"
        title = "\n".join(text.get_text() for text in figure.axes[0].texts)
        assert "Semantic specialist — L1 H24" in title
        assert "Net: $D_{\\rm rel} = +0.481" in title
        assert "J = 1.350" in title
        labels = "\n".join(
            text.get_text()
            for axis in figure.axes
            for text in axis.texts
            if "Graph-local" in text.get_text()
        )
        assert "D_{\\rm rel} = +0.481" in labels
        assert "D_{\\rm rel} = +0.102" in labels
        assert "D_{\\rm rel} = -0.221" in labels
        assert "J = 1.350" in labels
        assert "J = 0.930" in labels
        assert "J = 1.710" in labels
        assert "J = 1.110" in labels
        assert "J = 0.840" in labels
        assert labels.count("Graph-local") == 5
    finally:
        plt.close(figure)


def test_attention_grid_renders_exactly_four_configured_rows_in_order():
    pytest.importorskip("rdkit")
    attention = np.full((4, 4), 0.25)
    payload = select_attention_examples_for_grid(
        {
            "examples": [
                {
                    "dataset_index": index,
                    "smiles": "CCO",
                    "attention": {"semantic": attention},
                }
                for index in (0, 5, 80, 100, 200)
            ]
        },
        graph_indices=(200, 5, 100, 0),
    )
    coordinates = {
        index: {"D_rel": index / 1000.0, "J": 1.0}
        for index in (200, 5, 100, 0)
    }
    figure = plot_attention_grid(
        payload,
        role="semantic",
        head=(1, 24),
        per_graph_coordinates=coordinates,
        net_d_rel=0.4,
        net_joint_sensitivity=1.2,
    )
    try:
        labels = [
            text.get_text()
            for axis in figure.axes
            for text in axis.texts
            if "Graph-local" in text.get_text()
        ]
        assert len(labels) == 4
        assert [
            int(label.splitlines()[0].removeprefix("PCQM index "))
            for label in labels
        ] == [200, 5, 100, 0]
    finally:
        plt.close(figure)


def test_av_pca_identifies_role_and_head_metrics():
    rng = np.random.default_rng(17)
    payload = {
        "head": (7, 14),
        "vectors": rng.normal(size=(12, 8)),
        "labels": ["other/diffuse"] * 12,
        "n_used": 12,
    }
    figure = plot_av_pca(
        payload,
        title_label="Structural specialist",
        d_rel=-0.275,
        joint_sensitivity=1.234,
    )
    try:
        title = figure.axes[0].get_title()
        assert title.startswith("PCA of head output")
        assert "Pooled" not in title
        assert "Structural specialist — L7 H14" in title
        assert "D_{\\rm rel} = -0.275" in title
        assert "J = 1.234" in title
    finally:
        plt.close(figure)


def test_av_pca_focus_palette_is_stable_across_plots():
    rng = np.random.default_rng(23)
    labels_a = (
        ["Ring: aromatic"] * 5
        + ["Ring: aliphatic"] * 2
        + ["O: carbonyl"] * 4
        + ["O: hydroxyl"] * 2
        + ["N: amide"] * 3
        + ["other/diffuse"] * 2
    )
    labels_b = (
        ["other/diffuse"] * 5
        + ["N: amide"] * 4
        + ["O: carbonyl"] * 3
        + ["Ring: junction"] * 2
    )

    def make_figure(labels):
        return plot_av_pca(
            {
                "head": (1, 24),
                "vectors": rng.normal(size=(len(labels), 8)),
                "labels": labels,
                "n_used": len(labels),
            },
            minimum_count=1,
        )

    first = make_figure(labels_a)
    second = make_figure(labels_b)
    try:
        def plotted_colors(figure):
            return {
                collection.get_label().split(" (n=", 1)[0]: tuple(
                    collection.get_facecolors()[0]
                )
                for collection in figure.axes[0].collections
                if collection.get_label() and collection.get_facecolors().size
            }

        first_colors = plotted_colors(first)
        second_colors = plotted_colors(second)
        for label in ("O: carbonyl", "N: amide", "other/diffuse"):
            assert np.allclose(first_colors[label], second_colors[label])
        assert not np.allclose(
            first_colors["Ring: aromatic"],
            second_colors["Ring: junction"],
        )
        assert not np.allclose(
            first_colors["Ring: aromatic"],
            first_colors["Ring: aliphatic"],
        )
        assert not np.allclose(
            first_colors["O: carbonyl"],
            first_colors["O: hydroxyl"],
        )
        for label in (
            "Ring: aromatic",
            "Ring: aliphatic",
            "Ring: junction",
            "O: carbonyl",
            "O: hydroxyl",
        ):
            expected = PCA_FOCUS_COLORS[label]
            actual = (
                first_colors[label]
                if label in first_colors
                else second_colors[label]
            )
            assert np.allclose(actual, to_rgba(expected, alpha=0.78))
        assert np.allclose(
            first_colors["other/diffuse"],
            to_rgba("#B8C2CA", alpha=0.78),
        )
        assert not np.allclose(
            first_colors["Ring: aromatic"],
            first_colors["other/diffuse"],
        )
    finally:
        plt.close(first)
        plt.close(second)


def test_layer_av_pca_collection_reuses_one_forward_per_graph(monkeypatch):
    num_layers = 3
    num_heads = 4
    num_nodes = 3
    head_width = 2
    graphs = [
        SimpleNamespace(smiles=f"graph-{index}", num_nodes=num_nodes)
        for index in range(3)
    ]
    model_layers = [
        SimpleNamespace(self_attn=SimpleNamespace(num_heads=num_heads))
        for _ in range(num_layers)
    ]
    figure_runtime = SimpleNamespace(
        backend=SimpleNamespace(
            model=SimpleNamespace(
                encoder=SimpleNamespace(
                    graph_encoder=SimpleNamespace(layers=model_layers)
                )
            )
        ),
        runtime=SimpleNamespace(eval_ds=graphs),
    )
    captured = SimpleNamespace(
        transport=tuple(
            torch.arange(
                num_heads * (num_nodes + 1) * head_width,
                dtype=torch.float32,
            ).reshape(num_heads, num_nodes + 1, head_width)
            + layer
            for layer in range(num_layers)
        ),
        attention=tuple(
            torch.ones(
                num_heads,
                num_nodes + 1,
                num_nodes + 1,
                dtype=torch.float32,
            )
            for _ in range(num_layers)
        ),
    )
    calls = []

    class FakeExtractor:
        def __init__(self, backend):
            assert backend is figure_runtime.backend

        def extract(self, graph):
            calls.append(graph.smiles)
            return captured

    monkeypatch.setattr(
        "graph_specialisation_metrics.methodology.graphormer_figure_data."
        "GraphormerDiagnosticExtractor",
        FakeExtractor,
    )
    monkeypatch.setattr(
        "graph_specialisation_metrics.methodology.graphormer_figure_data."
        "_attention_focus_categories",
        lambda smiles: ["Ring: aromatic"] * num_nodes,
    )
    payload = compute_layer_av_pca_inputs(
        figure_runtime,
        layers=(0, 2),
        n_graphs=3,
        verbose=False,
    )

    assert calls == ["graph-0", "graph-1", "graph-2"]
    assert payload["requested_layers"] == (0, 2)
    assert payload["num_heads"] == num_heads
    assert payload["n_used"] == 3
    for layer in (0, 2):
        layer_payload = payload["layers"][layer]
        assert layer_payload["vectors"].shape == (3, num_heads, head_width)
        assert np.asarray(layer_payload["labels"]).shape == (3, num_heads)
        assert {
            label
            for row in layer_payload["labels"]
            for label in row
        } == {"Ring: aromatic"}


def test_layer_av_pca_grid_is_4x8_with_readable_legend_below():
    rng = np.random.default_rng(71)
    categories = list(PCA_FOCUS_COLORS)[:18]
    num_graphs = 36
    payload = {
        "layer": 1,
        "vectors": rng.normal(size=(num_graphs, 32, 8)),
        "labels": [
            [
                categories[(graph + head) % len(categories)]
                for head in range(32)
            ]
            for graph in range(num_graphs)
        ],
        "n_used": num_graphs,
    }
    figure = plot_layer_av_pca_grid(payload)
    try:
        figure.canvas.draw()
        assert len(figure.axes) == 32
        assert figure.axes[0].get_title().startswith("H0\n")
        assert figure.axes[-1].get_title().startswith("H31\n")
        assert {
            (
                axis.get_subplotspec().rowspan.start,
                axis.get_subplotspec().colspan.start,
            )
            for axis in figure.axes
        } == {(row, column) for row in range(4) for column in range(8)}

        assert len(figure.legends) == 1
        legend = figure.legends[0]
        assert [text.get_text() for text in legend.get_texts()] == list(
            PCA_FOCUS_COLORS
        )
        assert all(
            not collection.get_rasterized()
            for axis in figure.axes
            for collection in axis.collections
        )
        renderer = figure.canvas.get_renderer()
        legend_box = legend.get_window_extent(renderer)
        axes_bottom = min(
            axis.get_window_extent(renderer).y0 for axis in figure.axes
        )
        assert 0 <= legend_box.x0 < legend_box.x1 <= figure.bbox.width
        assert 0 <= legend_box.y0 < legend_box.y1 < axes_bottom
    finally:
        plt.close(figure)


def test_graphormer_diagnostic_extractor_matches_exact_attention_sites():
    transformers = pytest.importorskip("transformers")
    config = transformers.GraphormerConfig(
        num_hidden_layers=2,
        embedding_dim=32,
        ffn_embedding_dim=32,
        num_attention_heads=4,
        num_classes=1,
        num_atoms=4608,
        num_edges=1536,
        num_in_degree=512,
        num_out_degree=512,
        num_spatial=512,
        num_edge_dis=128,
        multi_hop_max_dist=5,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
    )
    model = transformers.GraphormerForGraphClassification(config).eval()
    data = tiny_graph(config)
    task = get_task("graphormer_pcqm4mv2")
    runtime = GraphormerRuntime(
        model,
        [data],
        [data],
        device=torch.device("cpu"),
        seed=0,
        metric_fn=task.metric_fn,
    )
    backend = GraphormerBackend(runtime, task, sigma=[1.0])
    captured = GraphormerDiagnosticExtractor(backend).extract(data)

    assert len(captured.dot) == config.num_hidden_layers
    assert captured.dot[0].shape == (
        config.num_attention_heads,
        data.num_nodes + 1,
        data.num_nodes + 1,
    )
    assert captured.transport[0].shape == (
        config.num_attention_heads,
        data.num_nodes + 1,
        config.embedding_dim // config.num_attention_heads,
    )
    expected_attention = torch.softmax(captured.dot[0] + captured.bias[0], dim=-1)
    assert torch.allclose(
        captured.attention[0], expected_attention, atol=1e-7, rtol=0.0
    )

    canonical = backend.capture([data], require_grad=False)
    assert torch.allclose(
        captured.transport[0],
        canonical.transport[0][0].permute(1, 0, 2),
        atol=1e-7,
        rtol=0.0,
    )
