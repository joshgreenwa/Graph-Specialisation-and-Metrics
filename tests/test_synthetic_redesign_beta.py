from __future__ import annotations

import importlib.util
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "experiments/synthetic/analysis/paper_synthetic_mixed_task_redesign_beta_colab.py"
)


def load_beta_module():
    name = "_paper_synthetic_mixed_task_redesign_beta_test"
    spec = importlib.util.spec_from_file_location(name, SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_four_reductions_are_padding_invariant_and_ordered():
    beta = load_beta_module()
    generator = torch.Generator().manual_seed(7)
    q = torch.randn(4, 5, 3, 2, 6, 4, generator=generator)
    valid = torch.tensor(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0],
            [1, 1, 0, 0, 0],
        ],
        dtype=torch.bool,
    )
    original = beta.aggregate_projected_events(q, valid)["per_graph"]
    padding = torch.randn(4, 3, 3, 2, 6, 4, generator=generator) * 1.0e4
    padded = beta.aggregate_projected_events(
        torch.cat((q, padding), dim=1),
        torch.cat((valid, torch.zeros(4, 3, dtype=torch.bool)), dim=1),
    )["per_graph"]
    for name in beta.AGGREGATIONS:
        torch.testing.assert_close(original[name], padded[name])
    assert torch.all(original["CG"] <= original["EG"] + 1.0e-5)
    assert torch.all(original["CN"] <= original["EN"] + 1.0e-5)
    assert torch.all(original["EN"] <= original["EG"] + 1.0e-5)


def test_distance_profiles_use_common_carrier_weighting_and_nan_without_support():
    beta = load_beta_module()
    q = torch.zeros(1, 2, 1, 1, 2, 1)
    q[0, :, 0, 0, 0, 0] = torch.tensor([1.0, 3.0])
    q[0, 0, 0, 0, 1, 0] = 2.0
    valid = torch.ones(1, 2, dtype=torch.bool)
    distance = torch.tensor([[[0, 0], [0, 1]]])
    profile = beta.distance_profiles(q, valid, distance, max_distance=2)
    # d=0: carrier 0 has event mean/sensitivity 2; carrier 1 has one event of 2.
    assert profile["F_sens"][0, 0, 0, 0].item() == pytest.approx(2.0)
    assert profile["F_coh"][0, 0, 0, 0].item() == pytest.approx(2.0)
    assert torch.isnan(profile["F_sens"][0, 0, 0, 2])
    assert torch.isnan(profile["F_coh"][0, 0, 0, 2])


def test_cross_graph_mismatch_never_uses_an_event_duplicate_from_same_graph():
    beta = load_beta_module()
    graph_ids = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    nodes = 3
    value = torch.arange(len(graph_ids) * nodes, dtype=torch.float32).reshape(-1, 1, 1)
    result = beta._cross_graph_mismatch_transport(value, graph_ids, nodes).reshape(
        len(graph_ids), nodes
    )
    original = value.reshape(len(graph_ids), nodes)
    for row in range(len(graph_ids)):
        assert not torch.equal(result[row], original[row])
        assert torch.equal(result[row], result[(row // 2) * 2])


def test_signed_causal_coordinates_retain_anti_alignment():
    beta = load_beta_module()
    causal = beta._signed_causal_coordinates(
        np.asarray([[1.0, -2.0]]), np.asarray([[-3.0, 4.0]])
    )
    assert causal["anti_aligned"].all()
    assert np.max(np.abs(causal["D"])) <= 1.0 + 1.0e-12
    assert np.all(causal["J"] >= 0)
    assert np.all(causal["G"] >= 0)


def test_specialist_families_can_be_empty_and_cache_identity_is_fail_closed():
    beta = load_beta_module()
    per_graph = np.ones((24, 2, 3), dtype=float)
    config = beta.BetaConfig(
        family_size=2,
        bootstrap_samples=20,
        causal_graphs=6,
        causal_events=1,
        causal_batch_size=6,
    )
    groups, diagnostics = beta.select_beta_families(
        per_graph, per_graph, config, seed=19
    )
    assert groups["semantic_specialist"] == []
    assert groups["structural_specialist"] == []
    assert groups["low_J_inert"] == []
    assert "semantic_specialist" in diagnostics["empty_families"]
    payload = {
        "version": beta.BETA_VERSION,
        "schema": beta.BETA_SCHEMA,
        "fingerprint": config.fingerprint,
        "checkpoint_sha256": "abc",
    }
    assert beta._beta_cache_is_current(payload, config, "abc")
    assert not beta._beta_cache_is_current(payload, config, "different")


def test_checkpoint_discovery_prefers_canonical_paper_fingerprint(tmp_path):
    beta = load_beta_module()
    repository = SOURCE.parents[3]
    legacy = beta.load_legacy_module(repository)
    checkpoint_dir = tmp_path / "cycle_dual_v2" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    canonical = legacy.Config()
    exploratory = replace(canonical, steps=canonical.steps + 1)
    for config in (canonical, exploratory):
        fingerprint = legacy.config_fingerprint(config)
        torch.save(
            {
                "version": beta.LEGACY_VERSION,
                "seed": 0,
                "config": asdict(config),
                "fingerprint": fingerprint,
            },
            checkpoint_dir / f"seed_0__{fingerprint}.pt",
        )
    path, payload = beta.discover_checkpoint(legacy, checkpoint_dir.parent, 0)
    canonical_fingerprint = legacy.config_fingerprint(canonical)
    assert payload["fingerprint"] == canonical_fingerprint
    assert path.name == f"seed_0__{canonical_fingerprint}.pt"


def test_cg_regression_manifest_uses_cuda_tolerant_structured_gate():
    beta = load_beta_module()
    comparison = beta._tensor_equivalence_diagnostic(
        torch.ones(8),
        torch.ones(8) + 3.05e-5,
        atol=beta.CG_REGRESSION_ATOL,
    )
    assert comparison["passed"]
    assert not beta._tensor_equivalence_diagnostic(
        torch.ones(8),
        torch.ones(8) + 1.0e-2,
        atol=beta.CG_REGRESSION_ATOL,
    )["passed"]
    assert beta._cg_formula_regression_passed(
        {
            "passed": True,
            "max_abs_error": 3.05e-5,
            "atol": beta.CG_REGRESSION_ATOL,
            "rtol": beta.CUDA_EQUIVALENCE_RTOL,
        }
    )
    assert beta._cg_formula_regression_passed(3.05e-5)  # legacy scalar manifest
    assert not beta._cg_formula_regression_passed({"passed": False})
    assert not beta._cg_formula_regression_passed(1.0e-2)
