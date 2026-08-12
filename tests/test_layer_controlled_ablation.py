from __future__ import annotations

import csv
import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest
from pypdf import PdfReader

from graph_specialisation_metrics.methodology.cache import CacheContract
from graph_specialisation_metrics.methodology.layer_controlled_ablation import (
    CacheDiscoveryError,
    HeadAblationData,
    association_summary,
    discover_cache_candidates,
    graph_bootstrap_interval,
    load_candidate,
    mean_within_layer_spearman,
    run_layer_controlled_correction,
    select_cache_candidate,
    stratified_residual_rank_spearman,
    verify_figure_bundle,
    within_layer_permutation_test,
)
from graph_specialisation_metrics.methodology.layer_controlled_ablation_figures import (
    MOLECULAR_FIGSIZE,
    SYNTHETIC_FIGSIZE,
    build_molecular_dissertation_panel,
    build_synthetic_dissertation_panel,
    render_molecular_dissertation_panel,
    render_molecular_within_layer_rank_panel,
    render_synthetic_dissertation_panel,
    render_synthetic_within_layer_rank_panel,
    within_stratum_percentile_ranks,
)
from graph_specialisation_metrics.methodology.protocol import PROTOCOL_VERSION, stable_hash


def _data(
    *,
    task: str = "synthetic",
    seed: int = 0,
    layers: int = 3,
    heads: int = 8,
    impact: np.ndarray | None = None,
    graphs: int | None = None,
) -> HeadAblationData:
    layer = np.repeat(np.arange(layers), heads)
    head_ids = tuple((layer_index, head) for layer_index in range(layers) for head in range(heads))
    within = np.tile(np.linspace(0.1, 1.0, heads), layers)
    joint = within + 4.0 * layer
    values = within + 10.0 * layer if impact is None else np.asarray(impact, dtype=np.float64)
    matrix = None
    graph_ids = None
    if graphs is not None:
        offsets = np.linspace(-0.01, 0.01, graphs)[:, None]
        matrix = values[None, :] + offsets
        graph_ids = np.arange(graphs)
    return HeadAblationData(
        task=task,
        seed=seed,
        joint_sensitivity=joint,
        layers=layer,
        head_ids=head_ids,
        ablation_impact=values,
        graph_ids=graph_ids,
        movement_by_graph=matrix,
    ).validate(expected_geometry=(layers, heads), strict=False)


def test_layer_controlled_effect_removes_simpson_confound():
    # Later layers have much larger J but much smaller impact. The pooled effect is
    # negative even though heads are perfectly ordered within every layer.
    data = _data()
    data.ablation_impact = np.tile(np.linspace(0.1, 1.0, 8), 3) - 10.0 * data.layers
    summary, layers = association_summary((data,))
    assert summary["pooled_raw_rho"] < 0
    assert summary["within_layer_mean_rho"] == pytest.approx(1.0)
    assert set(summary["leave_one_layer_out"]) == {"0", "1", "2"}
    assert all(value == pytest.approx(1.0) for value in summary["leave_one_layer_out"].values())
    assert len(layers) == 3
    assert all(row["rho"] == pytest.approx(1.0) for row in layers)


def test_equal_layer_mean_matches_centred_rank_estimator_without_ties():
    data = _data()
    estimate, _rows = mean_within_layer_spearman(data)
    assert estimate == pytest.approx(1.0)
    assert stratified_residual_rank_spearman((data,)) == pytest.approx(estimate)


def test_tied_average_ranks_are_supported():
    from scipy.stats import spearmanr

    data = _data(layers=2, heads=4)
    data.joint_sensitivity = np.tile((1.0, 1.0, 2.0, 3.0), 2)
    data.ablation_impact = np.tile((1.0, 2.0, 2.0, 3.0), 2)
    expected = float(spearmanr(data.joint_sensitivity[:4], data.ablation_impact[:4]).statistic)
    estimate, rows = mean_within_layer_spearman(data)
    assert estimate == pytest.approx(expected)
    assert all(row["rho"] == pytest.approx(expected) for row in rows)
    assert stratified_residual_rank_spearman((data,)) == pytest.approx(expected)


def test_seed_weighting_is_explicit():
    first = _data(seed=0)
    second = _data(seed=1)
    second.ablation_impact = -second.ablation_impact
    # One seed is positive, the other negative; the trained-seed mean is zero.
    summary, _rows = association_summary((first, second))
    assert summary["seed_estimates"][0]["within_layer_mean_rho"] == pytest.approx(1.0)
    assert summary["seed_estimates"][1]["within_layer_mean_rho"] == pytest.approx(-1.0)
    assert summary["within_layer_mean_rho"] == pytest.approx(0.0, abs=1e-12)


def test_synthetic_independent_within_layer_regression_rounds_to_point_78():
    higher = np.asarray((0, 1, 2, 3, 6, 7, 5, 4), dtype=np.float64)
    lower = np.asarray((0, 1, 2, 3, 7, 6, 5, 4), dtype=np.float64)
    datasets = []
    stratum = 0
    for seed in range(3):
        impacts = []
        for _layer in range(3):
            impacts.extend(lower if stratum < 2 else higher)
            stratum += 1
        data = _data(seed=seed)
        data.joint_sensitivity = np.tile(np.arange(8, dtype=np.float64), 3)
        data.ablation_impact = np.asarray(impacts)
        datasets.append(data)
    summary, _rows = association_summary(datasets)
    assert round(summary["within_layer_mean_rho"], 2) == 0.78
    assert round(summary["stratified_residual_rank_rho"], 2) == 0.78


def test_graph_bootstrap_is_reproducible_and_resamples_shared_graphs():
    data = _data(task="zinc", seed=42, layers=2, heads=4, graphs=12)
    first = graph_bootstrap_interval((data,), replicates=50, rng_seed=7)
    second = graph_bootstrap_interval((data,), replicates=50, rng_seed=7)
    assert first == second
    assert first["estimate"] == pytest.approx(1.0)
    assert first["resampled_unit"].startswith("held-out molecule")


def test_within_layer_permutation_is_reproducible():
    data = _data(layers=2, heads=5)
    first = within_layer_permutation_test((data,), replicates=250, rng_seed=11)
    second = within_layer_permutation_test((data,), replicates=250, rng_seed=11)
    assert first == second
    assert first["observed"] == pytest.approx(1.0)
    assert first["shuffle_strata"] == ["trained_seed", "layer"]


def test_validate_rejects_wrong_geometry_and_incomplete_graph_matrix():
    with pytest.raises(ValueError, match="observed geometry"):
        _data(layers=2, heads=4).validate(expected_geometry=(3, 4))
    data = _data(task="zinc", layers=2, heads=4)
    with pytest.raises(ValueError, match="graph-level rows are required"):
        data.validate(expected_geometry=(2, 4), expected_graphs=12, strict=True)


def test_discovery_prefers_exact_cache_and_requires_opt_in_for_population(tmp_path):
    metrics = tmp_path / "graph_specialisation_metrics"
    _write_canonical_cache(metrics, "zinc", graphs=64, target_pooled_rho=0.80)
    _write_population_core(metrics, "zinc", graphs=4)
    exact = metrics / "canonical_methodology_v4_zinc_qm9/zinc/seed_42/cache"

    candidates = discover_cache_candidates(metrics)
    selected = select_cache_candidate("zinc", candidates)
    assert selected.family == "canonical_validation"
    assert selected.exact_dissertation

    exact.joinpath("scores/raw.pt").unlink()
    exact.joinpath("causal/validation.pt").unlink()
    candidates = discover_cache_candidates(metrics)
    with pytest.raises(CacheDiscoveryError, match="no complete exact dissertation cache"):
        select_cache_candidate("zinc", candidates)
    fallback = select_cache_candidate(
        "zinc",
        candidates,
        allow_non_dissertation_fallbacks=True,
        strict=False,
    )
    assert fallback.family == "population_core"
    assert not fallback.exact_dissertation


def test_historical_direct_synthetic_shards_load_without_canonical_wrapper(tmp_path):
    import torch

    metrics = tmp_path / "graph_specialisation_metrics"
    analysis = metrics / "causal_specialisation_double_dissociation/cycle_dual_v2/analysis"
    analysis.mkdir(parents=True)
    base = np.arange(1, 25, dtype=np.float64).reshape(3, 8)
    functional = np.repeat(base[..., None], 4, axis=-1)
    for seed in range(3):
        torch.save(
            {
                "version": ("causal-specialisation-double-dissociation-v2-shared-source-marker"),
                "analysis_version": "causal-specialisation-matched-donor-swaps-v1",
                "fingerprint": "training-fingerprint",
                "analysis_fingerprint": "5a29d56ac5d21586",
                "seed": seed,
                "semantic_score": base + seed,
                "structural_score": base + seed,
                "ablation_semantic": {"functional": functional + seed},
                "ablation_structural": {"functional": functional + seed},
            },
            analysis / f"seed_{seed}__5a29d56ac5d21586.pt",
        )

    candidate = select_cache_candidate(
        "synthetic",
        discover_cache_candidates(metrics),
        strict=False,
    )
    datasets = load_candidate(candidate, strict=False)
    assert [data.seed for data in datasets] == [0, 1, 2]
    assert all(len(data.head_ids) == 24 for data in datasets)
    assert all(data.cache_contract["validated_shard"] is not None for data in datasets)


def test_shallow_discovery_refuses_equally_preferred_exact_roots(tmp_path):
    metrics = tmp_path / "graph_specialisation_metrics"
    for parent in ("relocated_a", "relocated_b"):
        _write_canonical_seed_root(
            metrics / parent / "zinc/seed_42",
            "zinc",
            graphs=64,
            target_pooled_rho=0.80,
        )
    candidates = discover_cache_candidates(metrics)
    with pytest.raises(CacheDiscoveryError, match="ambiguous compatible caches"):
        select_cache_candidate("zinc", candidates)


def test_incomplete_known_cache_cannot_hide_complete_relocated_cache(tmp_path):
    metrics = tmp_path / "graph_specialisation_metrics"
    _write_graphormer_cache(metrics, graphs=2, target_pooled_rho=0.53)
    relocated = metrics / "relocated/graphormer_pcqm4mv2/seed_0"
    _write_graphormer_cache(
        metrics,
        graphs=128,
        seed_root=relocated,
        target_pooled_rho=0.53,
    )

    selected = select_cache_candidate(
        "graphormer_pcqm4mv2",
        discover_cache_candidates(metrics),
    )
    assert selected.root == relocated
    assert selected.origin == "shallow Drive search"


def test_cache_contract_task_identity_fails_closed(tmp_path):
    metrics = tmp_path / "graph_specialisation_metrics"
    cache = metrics / "canonical_methodology_v4_zinc_qm9/zinc/seed_42/cache"
    _write_wrapped(
        cache / "scores/raw.pt",
        {"coordinates": {"joint_sensitivity": np.ones((10, 8))}},
        task="qm9_gap_dense",
        seed=42,
        geometry={"layers": 10, "heads": 8},
    )
    _write_wrapped(
        cache / "causal/validation.pt",
        {"clean_ablation": {}},
        task="qm9_gap_dense",
        seed=42,
        geometry={"layers": 10, "heads": 8},
    )
    with pytest.raises(CacheDiscoveryError, match="cache task"):
        select_cache_candidate("zinc", discover_cache_candidates(metrics))


def test_paired_scientific_contract_mismatch_fails_before_selection(tmp_path):
    import torch

    metrics = tmp_path / "graph_specialisation_metrics"
    _write_canonical_cache(metrics, "zinc", graphs=64)
    causal = metrics / "canonical_methodology_v4_zinc_qm9/zinc/seed_42/cache/causal/validation.pt"
    value = torch.load(causal, map_location="cpu", weights_only=False)["value"]
    _write_wrapped(
        causal,
        value,
        task="zinc",
        seed=42,
        geometry={"layers": 10, "heads": 8},
        event_manifest_hash="e" * 64,
        protocol_fingerprint="different-scientific-protocol",
    )
    with pytest.raises(CacheDiscoveryError, match="protocol_fingerprint"):
        select_cache_candidate("zinc", discover_cache_candidates(metrics))


def _synthetic_records():
    rows = []
    for seed, marker_shift in enumerate((1.0, 1.03, 0.97)):
        for layer in range(3):
            for head in range(8):
                x = marker_shift * 10 ** (-2.0 + layer + 0.08 * head)
                rows.append(
                    {
                        "seed": seed,
                        "layer": layer,
                        "joint_sensitivity": x,
                        "ablation_impact": 0.2 + 0.6 * x,
                    }
                )
    return rows


def _molecular_record():
    joint = np.linspace(0.2, 3.0, 80)
    return {
        "seed": 42,
        "joint_sensitivity": joint,
        "ablation_impact": 0.01 + 0.03 * joint,
        "layer": np.repeat(np.arange(10), 8),
    }


def test_percentile_rank_transform_is_tie_aware_and_stratified():
    transformed = within_stratum_percentile_ranks(
        (1.0, 2.0, 3.0, 100.0, 100.0, 300.0),
        ((0, 0), (0, 0), (0, 0), (0, 1), (0, 1), (0, 1)),
    )
    assert transformed[:3] == pytest.approx((1 / 6, 1 / 2, 5 / 6))
    assert transformed[3:] == pytest.approx((1 / 3, 1 / 3, 5 / 6))


def test_figure_artists_preserve_dissertation_grammar_in_rank_view():
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    raw_synthetic = build_synthetic_dissertation_panel(
        _synthetic_records(),
        estimate=0.78,
        pooled_estimate=0.77,
    )
    rank_synthetic = build_synthetic_dissertation_panel(
        _synthetic_records(),
        estimate=0.78,
        pooled_estimate=0.77,
        coordinate_view="within_layer_percentile_ranks",
    )
    raw_molecular = build_molecular_dissertation_panel(
        _molecular_record(),
        estimate=0.66,
        pooled_estimate=0.53,
        low=0.50,
        high=0.78,
    )
    rank_molecular = build_molecular_dissertation_panel(
        _molecular_record(),
        estimate=0.66,
        pooled_estimate=0.53,
        low=0.50,
        high=0.78,
        coordinate_view="within_layer_percentile_ranks",
    )
    try:
        raw_axis = raw_synthetic.axes[0]
        rank_axis = rank_synthetic.axes[0]
        assert rank_axis.texts[0].get_text().splitlines() == [
            r"$\rho$ = 0.77",
            r"Within-layer $\bar{\rho}$ = 0.78",
        ]
        assert (raw_axis.get_xscale(), raw_axis.get_yscale()) == ("log", "log")
        assert (rank_axis.get_xscale(), rank_axis.get_yscale()) == ("linear", "linear")
        assert rank_axis.get_xlim() == pytest.approx((0.0, 1.0))
        assert rank_axis.get_ylim() == pytest.approx((0.0, 1.0))
        assert "percentile" in rank_axis.get_xlabel().lower()
        assert "percentile" in rank_axis.get_ylabel().lower()
        assert rank_axis.get_position().bounds == pytest.approx((0.275, 0.22, 0.65, 0.62))
        assert rank_axis.collections[0].get_alpha() == pytest.approx(0.90)
        assert mcolors.to_hex(rank_axis.collections[0].get_facecolors()[0]) == "#6b5b95"
        assert mcolors.to_hex(rank_axis.collections[8].get_facecolors()[0]) == "#2a8c82"
        rank_offsets = np.asarray(
            [collection.get_offsets()[0] for collection in rank_axis.collections]
        )
        expected_positions = (np.arange(8) + 0.5) / 8
        for seed in range(3):
            for layer in range(3):
                start = (seed * 3 + layer) * 8
                assert rank_offsets[start : start + 8, 0] == pytest.approx(expected_positions)

        raw_molecular_axis = raw_molecular.axes[0]
        rank_molecular_axis = rank_molecular.axes[0]
        assert rank_molecular_axis.texts[0].get_text().splitlines() == [
            r"$\rho$ = 0.53",
            r"Within-layer $\bar{\rho}$ = 0.66  [0.50, 0.78]",
        ]
        assert raw_molecular_axis.get_xlabel() == r"Joint sensitivity, $J$"
        assert "percentile" in rank_molecular_axis.get_xlabel().lower()
        assert "percentile" in rank_molecular_axis.get_ylabel().lower()
        assert rank_molecular_axis.get_xlim() == pytest.approx((0.0, 1.0))
        assert rank_molecular_axis.get_ylim() == pytest.approx((0.0, 1.0))
        assert rank_molecular_axis.get_position().bounds == pytest.approx((0.17, 0.25, 0.56, 0.66))
        molecular_scatter = rank_molecular_axis.collections[0]
        assert molecular_scatter.get_alpha() == pytest.approx(0.86)
        assert molecular_scatter.get_cmap().name == "viridis"
        assert molecular_scatter.get_cmap().N == 10
        assert rank_molecular.legends[0]._ncols == 4
        molecular_offsets = np.asarray(molecular_scatter.get_offsets())
        for layer in range(10):
            assert molecular_offsets[layer * 8 : (layer + 1) * 8, 0] == pytest.approx(
                expected_positions
            )
    finally:
        for figure in (
            raw_synthetic,
            rank_synthetic,
            raw_molecular,
            rank_molecular,
        ):
            plt.close(figure)


def test_dissertation_figure_canvases_and_statistic_labels(tmp_path):
    synthetic = render_synthetic_dissertation_panel(
        _synthetic_records(),
        estimate=0.78,
        pooled_estimate=0.77,
        output_dir=tmp_path / "synthetic",
        metadata={"test": True},
    )
    synthetic_check = verify_figure_bundle(synthetic, expected_inches=SYNTHETIC_FIGSIZE)
    assert synthetic_check["png_pixels"] == [1306, 1416]

    molecular = render_molecular_dissertation_panel(
        _molecular_record(),
        estimate=0.66,
        pooled_estimate=0.53,
        low=0.50,
        high=0.78,
        output_dir=tmp_path / "zinc",
        metadata={"test": True},
    )
    molecular_check = verify_figure_bundle(molecular, expected_inches=MOLECULAR_FIGSIZE)
    assert molecular_check["png_pixels"] == [2520, 2130]

    for paths, points in ((synthetic, (156.72, 169.92)), (molecular, (302.4, 255.6))):
        pdf = next(Path(path) for path in paths if Path(path).suffix == ".pdf")
        page = PdfReader(str(pdf)).pages[0]
        assert float(page.mediabox.width) == pytest.approx(points[0], abs=0.02)
        assert float(page.mediabox.height) == pytest.approx(points[1], abs=0.02)
        assert b"/Subtype /Type3" not in pdf.read_bytes()


def test_rank_figure_exports_are_primary_and_self_describing(tmp_path):
    synthetic = render_synthetic_within_layer_rank_panel(
        _synthetic_records(),
        estimate=0.78,
        pooled_estimate=0.77,
        output_dir=tmp_path / "synthetic",
        metadata={"test": True},
    )
    molecular = render_molecular_within_layer_rank_panel(
        _molecular_record(),
        estimate=0.66,
        pooled_estimate=0.53,
        low=0.50,
        high=0.78,
        output_dir=tmp_path / "zinc",
        metadata={"test": True},
    )
    assert Path(synthetic[0]).stem == "fig2a_sensitivity_impact_within_layer_ranks"
    assert Path(molecular[0]).stem == "01_joint_sensitivity_head_ablation_within_layer_ranks"
    for paths in (synthetic, molecular):
        sidecar = next(Path(path) for path in paths if Path(path).name.endswith(".metadata.json"))
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        assert payload["coordinate_view"] == "within_layer_percentile_ranks"
        assert payload["rank_strata"] == ["trained_seed", "layer"]
        assert payload["role"] == "primary layer-confound-free diagnostic"
    synthetic_sidecar = json.loads(Path(synthetic[2]).read_text(encoding="utf-8"))
    assert synthetic_sidecar["pdf_raster_fallback_dpi"] == 600


def test_colab_builder_keeps_frontend_in_sync(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    builder = repository / "experiments/methodology/build_layer_controlled_head_ablation_colab.py"
    source = repository / "experiments/methodology/layer_controlled_head_ablation_colab.py"
    output = tmp_path / "launcher.ipynb"
    subprocess.run(
        [sys.executable, str(builder), "--source", str(source), "--output", str(output)],
        check=True,
    )
    notebook = json.loads(output.read_text(encoding="utf-8"))
    assert notebook["metadata"]["accelerator"] == "CPU"
    code = "".join(notebook["cells"][1]["source"])
    assert '"graphbench_bipartite_matching_hard"' not in code
    assert '"graphormer_pcqm4mv2"' in code
    assert "model_forwards=0" in code


def _write_synthetic_cache(root: Path) -> None:
    path = (
        root / "causal_specialisation_double_dissociation/cycle_dual_v2/tables/per_head_metrics.csv"
    )
    path.parent.mkdir(parents=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "seed",
                "layer",
                "head",
                "joint_score_J",
                "ablation_joint_impact",
            ),
        )
        writer.writeheader()
        for seed in range(3):
            for layer in range(3):
                for head in range(8):
                    writer.writerow(
                        {
                            "seed": seed,
                            "layer": layer,
                            "head": head,
                            "joint_score_J": 0.2 + head + 2.0 * layer,
                            "ablation_joint_impact": 0.3 + head + 3.0 * layer,
                        }
                    )


def _write_wrapped(
    path: Path,
    value: object,
    *,
    task: str,
    seed: int,
    geometry: dict[str, int],
    event_manifest_hash: str = "c" * 64,
    protocol_fingerprint: str = "test-protocol",
) -> None:
    import torch

    contract = CacheContract(
        protocol_fingerprint=protocol_fingerprint,
        task=task,
        task_adapter_version="test-adapter-v1",
        checkpoint_sha256="a" * 64,
        train_seed=int(seed),
        model_geometry=dict(geometry),
        output_representation="test-output",
        sigma=(1.0,),
        split_fingerprint="b" * 64,
        event_manifest_hash=event_manifest_hash,
        donors_per_source=2,
        source_cap=3,
        bootstrap_seed=123,
        repository_commit="test-checkout",
        bootstrap_replicates=20,
    )
    record = dataclasses.asdict(contract)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "metadata": {
                "protocol_version": PROTOCOL_VERSION,
                "contract": record,
                "contract_fingerprint": contract.fingerprint,
                "provenance_fingerprint": stable_hash(record),
            },
            "value": value,
        },
        path,
    )


def _impact_with_rounded_spearman(joint: np.ndarray, target: float) -> np.ndarray:
    values = np.asarray(joint, dtype=np.float64).reshape(-1)
    x_ranks = np.argsort(np.argsort(values, kind="stable"), kind="stable")
    noise = np.random.default_rng(8128 + len(values)).normal(size=len(values))
    upper = max(1.0, float(np.ptp(values))) * 8.0
    for scale in np.linspace(0.0, upper, 20_001):
        candidate = values + scale * noise
        y_ranks = np.argsort(np.argsort(candidate, kind="stable"), kind="stable")
        rho = float(np.corrcoef(x_ranks, y_ranks)[0, 1])
        if round(rho, 2) == float(target):
            return candidate
    raise AssertionError(f"could not construct a mock Spearman rho rounding to {target:.2f}")


def _write_graphormer_cache(
    root: Path,
    *,
    graphs: int = 4,
    seed_root: Path | None = None,
    target_pooled_rho: float | None = None,
) -> None:
    seed_root = seed_root or (root / "graphormer_pcqm4mv2_causal/graphormer_pcqm4mv2/seed_0")
    layers = np.repeat(np.arange(12), 32)
    heads = tuple((layer, head) for layer in range(12) for head in range(32))
    joint = np.tile(np.linspace(0.1, 1.0, 32), 12) + 0.2 * layers
    impact = 0.01 + 0.02 * joint
    if target_pooled_rho is not None:
        impact = _impact_with_rounded_spearman(joint, target_pooled_rho)
    pooled_rho = float(
        np.corrcoef(np.argsort(np.argsort(joint)), np.argsort(np.argsort(impact)))[0, 1]
    )
    core = {
        "clean_ablation": {
            "head_order": heads,
            "J": joint,
            "layers": layers,
            "prediction_movement": impact,
            "spearman_rho": pooled_rho,
        }
    }
    core_path = seed_root / "cache/focused/core_tests.pt"
    geometry = {"layers": 12, "heads": 32}
    _write_wrapped(
        core_path,
        core,
        task="graphormer_pcqm4mv2",
        seed=0,
        geometry=geometry,
    )
    offsets = np.linspace(-0.001, 0.001, graphs)
    for graph, offset in enumerate(offsets):
        path = seed_root / f"cache/focused/clean_ablation/graph_{graph:06d}.pt"
        _write_wrapped(
            path,
            {
                "graph": graph,
                "rows": [
                    {
                        "graph": graph,
                        "head": head_id,
                        "prediction_movement": float(impact[position] + offset),
                    }
                    for position, head_id in enumerate(heads)
                ],
            },
            task="graphormer_pcqm4mv2",
            seed=0,
            geometry=geometry,
        )


def _write_canonical_seed_root(
    seed_root: Path,
    task: str,
    *,
    graphs: int = 4,
    target_pooled_rho: float | None = None,
) -> None:
    layers = np.repeat(np.arange(10), 8)
    head_ids = tuple((layer, head) for layer in range(10) for head in range(8))
    joint = np.tile(np.linspace(0.1, 1.0, 8), 10) + 0.2 * layers
    impact = 0.01 + 0.02 * joint
    if target_pooled_rho is not None:
        impact = _impact_with_rounded_spearman(joint, target_pooled_rho)
    pooled_rho = float(
        np.corrcoef(np.argsort(np.argsort(joint)), np.argsort(np.argsort(impact)))[0, 1]
    )
    score_path = seed_root / "cache/scores/raw.pt"
    geometry = {"layers": 10, "heads": 8}
    _write_wrapped(
        score_path,
        {"coordinates": {"joint_sensitivity": joint.reshape(10, 8)}},
        task=task,
        seed=42,
        geometry=geometry,
        event_manifest_hash="d" * 64,
    )
    offsets = np.linspace(-0.001, 0.001, graphs)
    clean = {}
    for position, head_id in enumerate(head_ids):
        clean[f"head_L{head_id[0]}_H{head_id[1]}"] = {
            "prediction_movement": float(impact[position]),
            "graphs": [
                {
                    "graph": graph,
                    "prediction_movement": float(impact[position] + offset),
                }
                for graph, offset in enumerate(offsets)
            ],
        }
    causal_path = seed_root / "cache/causal/validation.pt"
    _write_wrapped(
        causal_path,
        {
            "clean_ablation": clean,
            "associations": {"J_vs_clean_prediction_movement": {"pooled": {"rho": pooled_rho}}},
        },
        task=task,
        seed=42,
        geometry=geometry,
        event_manifest_hash="e" * 64,
    )


def _write_canonical_cache(
    root: Path,
    task: str,
    *,
    graphs: int = 4,
    target_pooled_rho: float | None = None,
) -> None:
    _write_canonical_seed_root(
        root / f"canonical_methodology_v4_zinc_qm9/{task}/seed_42",
        task,
        graphs=graphs,
        target_pooled_rho=target_pooled_rho,
    )


def _write_population_core(root: Path, task: str, *, graphs: int = 4) -> None:
    seed_root = root / f"grit_dense_causal_population_paper/{task}/seed_42"
    layers = np.repeat(np.arange(10), 8)
    heads = tuple((layer, head) for layer in range(10) for head in range(8))
    joint = np.tile(np.linspace(0.1, 1.0, 8), 10) + 0.2 * layers
    impact = 0.01 + 0.02 * joint
    geometry = {"layers": 10, "heads": 8}
    _write_wrapped(
        seed_root / "cache/focused_population/core_tests.pt",
        {
            "clean_ablation": {
                "head_order": heads,
                "J": joint,
                "layers": layers,
                "prediction_movement": impact,
                "spearman_rho": 1.0,
            }
        },
        task=task,
        seed=42,
        geometry=geometry,
    )
    offsets = np.linspace(-0.001, 0.001, graphs)
    for graph, offset in enumerate(offsets):
        _write_wrapped(
            seed_root / f"cache/focused/clean_ablation/graph_{graph:06d}.pt",
            {
                "graph": graph,
                "rows": [
                    {
                        "graph": graph,
                        "head": head_id,
                        "prediction_movement": float(impact[position] + offset),
                    }
                    for position, head_id in enumerate(heads)
                ],
            },
            task=task,
            seed=42,
            geometry=geometry,
        )


def test_mock_cache_end_to_end_is_model_free(tmp_path):
    metrics = tmp_path / "graph_specialisation_metrics"
    _write_synthetic_cache(metrics)
    _write_graphormer_cache(metrics)
    _write_canonical_cache(metrics, "zinc")
    _write_canonical_cache(metrics, "qm9_gap_dense")
    result = run_layer_controlled_correction(
        metrics_root=metrics,
        output_root=metrics / "correction",
        bootstrap_replicates=8,
        permutation_replicates=20,
        strict=False,
    )
    assert result["cache_only"] is True
    assert result["model_forwards"] == 0
    assert set(result["tasks"]) == {
        "synthetic",
        "graphormer_pcqm4mv2",
        "zinc",
        "qm9_gap_dense",
    }
    assert Path(result["manifest_path"]).is_file()
    assert Path(result["tables"]["pooled_vs_layer_controlled"]).is_file()
    for task in result["tasks"].values():
        assert task["summary"]["within_layer_mean_rho"] == pytest.approx(1.0)
        assert all(Path(path).is_file() for path in task["figures"])
