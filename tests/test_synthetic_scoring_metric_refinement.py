from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = (
    ROOT
    / "experiments"
    / "synthetic"
    / "analysis"
    / "scoring_metric_refinement_synthetic.py"
)
FRONTEND = (
    ROOT
    / "experiments"
    / "methodology"
    / "colab_scoring_metric_refinement_synthetic.py"
)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _runner():
    return _load(RUNNER, "synthetic_scoring_refinement_runner_test")


def _legacy(runner):
    return runner.load_legacy_module(ROOT)


def test_method_axes_match_predeclared_m1_m4_m5_m7_contract():
    from graph_specialisation_metrics.scoring_refinement.synthetic_validation import (
        SYNTHETIC_METHODS,
        method_score_axes,
    )

    shape = (2, 3)
    components = {
        "eg_semantic_single": np.full(shape, 1.0),
        "eg_semantic_transposition": np.full(shape, 2.0),
        "eg_pe_single": np.full(shape, 3.0),
        "eg_pe_transposition": np.full(shape, 4.0),
        "semantic_attention_follow": np.full(shape, 5.0),
        "semantic_attention_invariant": np.full(shape, 6.0),
        "pe_attention_follow": np.full(shape, 7.0),
        "pe_attention_invariant": np.full(shape, 8.0),
    }
    axes = method_score_axes(components)
    assert tuple(axes) == SYNTHETIC_METHODS
    expected = {
        "M1_DD": (1.0, 3.0),
        "M1_DT": (1.0, 4.0),
        "M1_TD": (2.0, 3.0),
        "M1_TT": (2.0, 4.0),
        "M4": (5.0, 6.0),
        "M5": (8.0, 7.0),
        "M7": (5.0, 7.0),
    }
    for method, pair in expected.items():
        assert float(axes[method][0][0, 0]) == pair[0]
        assert float(axes[method][1][0, 0]) == pair[1]


def test_synthetic_variants_change_only_the_declared_channel():
    torch = pytest.importorskip("torch")
    runner = _runner()
    legacy = _legacy(runner)
    cfg = legacy.Config(
        n=12,
        classes=6,
        key_vocab=16,
        rrwp_steps=6,
        dim=48,
        heads=4,
        layers=2,
        score_donors=2,
        seeds=(0,),
    )
    semantic_clean = legacy.make_batch(
        cfg, 3, seed=31, mode=legacy.MODE_SEMANTIC
    )
    semantic, semantic_partners = runner.make_variant_replicas(
        legacy,
        cfg,
        semantic_clean,
        variant="semantic_transposition",
        donors=2,
        seed=32,
    )
    for graph in range(len(semantic_clean)):
        clean_index = graph * 3
        for event in range(2):
            changed = clean_index + event + 1
            source = int(semantic_clean.target_idx[graph])
            partner = int(semantic_partners[graph, event])
            query = int(semantic_clean.q_idx[graph])
            assert partner not in {source, query}
            assert torch.equal(
                semantic.x[changed, source],
                semantic_clean.x[graph, partner],
            )
            assert torch.equal(
                semantic.x[changed, partner],
                semantic_clean.x[graph, source],
            )
            assert torch.equal(
                semantic.rrwp[changed], semantic_clean.rrwp[graph]
            )

    structural_clean = legacy.make_batch(
        cfg, 3, seed=41, mode=legacy.MODE_STRUCTURAL
    )
    for variant in ("pe_single_donor", "pe_transposition"):
        replicas, partners = runner.make_variant_replicas(
            legacy,
            cfg,
            structural_clean,
            variant=variant,
            donors=2,
            seed=42,
        )
        for graph in range(len(structural_clean)):
            clean_index = graph * 3
            for event in range(2):
                changed = clean_index + event + 1
                source = int(structural_clean.target_idx[graph])
                partner = int(partners[graph, event])
                assert torch.equal(
                    replicas.x[changed], structural_clean.x[graph]
                )
                expected = (
                    runner.copy_rrwp_footprint(
                        structural_clean.rrwp[graph], source, partner
                    )
                    if variant == "pe_single_donor"
                    else legacy.transpose_rrwp(
                        structural_clean.rrwp[graph], source, partner
                    )
                )
                assert torch.equal(replicas.rrwp[changed], expected)
                assert not torch.equal(
                    replicas.rrwp[changed], structural_clean.rrwp[graph]
                )


def test_appendix_attention_scoring_has_following_extreme():
    torch = pytest.importorskip("torch")
    runner = _runner()
    legacy = _legacy(runner)
    cfg = legacy.Config(
        n=3,
        classes=2,
        key_vocab=4,
        rrwp_steps=2,
        dim=6,
        heads=1,
        layers=1,
        score_donors=1,
        seeds=(0,),
    )
    clean = legacy.make_batch(
        cfg, 1, seed=51, mode=legacy.MODE_SEMANTIC
    )
    source = int(clean.target_idx[0])
    partner = next(
        node
        for node in range(cfg.n)
        if node not in {source, int(clean.q_idx[0])}
    )
    base = torch.full((cfg.n, cfg.n, cfg.heads), 0.1)
    base[source, :, 0] = torch.tensor([0.8, 0.7, 0.6])
    base[partner, :, 0] = torch.tensor([0.2, 0.3, 0.4])
    event = base.clone()
    event[source] = base[partner]
    event[partner] = base[source]
    attention = torch.stack([base, event], dim=0).reshape(-1, 1)
    follow, invariant = runner.appendix_attention_scores(
        attention,
        clean=clean,
        partners=np.asarray([[partner]], dtype=np.int64),
        cfg=cfg,
        temperature=0.1,
    )
    assert follow.shape == invariant.shape == (1, 1)
    assert float(follow[0, 0]) == pytest.approx(1.0)
    assert 0.0 < float(invariant[0, 0]) < 1.0


def _fake_result(seed: int):
    from graph_specialisation_metrics.scoring_refinement.synthetic_validation import (
        SYNTHETIC_METHODS,
    )

    layers, heads, graphs = 2, 3, 5
    base = np.arange(layers * heads, dtype=float).reshape(layers, heads)
    method_scores = {}
    selected = {}
    family = {}
    for index, method in enumerate(SYNTHETIC_METHODS):
        method_scores[method] = {
            "semantic": 0.5 + 0.04 * base + 0.01 * index,
            "structural": 0.8 - 0.025 * base + 0.01 * index,
        }
        selected[method] = {
            "semantic": [(0, 2), (1, 2)],
            "structural": [(0, 0), (1, 0)],
        }
        tasks = {}
        for task_index, task in enumerate(("semantic", "structural")):
            families = {}
            for family_index, family_name in enumerate(
                ("semantic", "structural")
            ):
                loss = np.zeros((3, graphs), dtype=float)
                loss[-1] = (
                    0.02
                    + 0.03 * float(task_index == family_index)
                    + 0.001 * index
                )
                families[family_name] = {"loss": loss}
            tasks[task] = {"families": families}
        family[method] = {"revision": f"test:{method}", "tasks": tasks}
    sem_ablation = np.repeat(
        (0.02 + 0.01 * base)[..., None], graphs, axis=-1
    )
    str_ablation = np.repeat(
        (0.07 - 0.006 * base)[..., None], graphs, axis=-1
    )
    sem_rescue = np.repeat(
        (0.01 + 0.02 * base)[..., None], graphs, axis=-1
    )
    str_rescue = np.repeat(
        (0.10 - 0.01 * base)[..., None], graphs, axis=-1
    )
    return {
        "seed": seed,
        "method_scores": method_scores,
        "method_selected_groups": selected,
        "method_family_ablation": family,
        "ablation_semantic": {"functional": sem_ablation},
        "ablation_structural": {"functional": str_ablation},
        "rescue_semantic": {"mediation": sem_rescue},
        "rescue_structural": {"mediation": str_rescue},
    }


def test_m1_arms_share_m1_dt_references_while_cosine_methods_self_calibrate():
    from graph_specialisation_metrics.scoring_refinement.synthetic_validation import (
        build_per_head_rows,
    )

    result = _fake_result(0)
    rows = build_per_head_rows([result])
    m1_dd = [row for row in rows if row["method"] == "M1_DD"]
    m4 = [row for row in rows if row["method"] == "M4"]
    expected = float(
        np.mean(result["method_scores"]["M1_DD"]["semantic"])
        / np.mean(result["method_scores"]["M1_DT"]["semantic"])
    )
    assert np.mean([row["semantic_score_norm"] for row in m1_dd]) == pytest.approx(
        expected
    )
    assert np.mean([row["semantic_score_norm"] for row in m4]) == pytest.approx(
        1.0
    )


def test_four_consolidated_figure_families_and_tables(tmp_path):
    pytest.importorskip("matplotlib")
    from graph_specialisation_metrics.scoring_refinement.synthetic_validation import (
        SYNTHETIC_METHODS,
        create_validation_outputs,
    )

    summary = create_validation_outputs(
        [_fake_result(0), _fake_result(1)], output_dir=tmp_path
    )
    assert set(summary["figures"]) == {
        "score_planes",
        "necessity_and_role",
        "rescue_role",
        "necessity_rescue",
    }
    for paths in summary["figures"].values():
        assert {Path(path).suffix for path in paths} == {".png", ".pdf"}
        assert all(Path(path).exists() for path in paths)
    assert len(summary["validation"]) == 3 * len(SYNTHETIC_METHODS)
    assert Path(summary["tables"]["per_head"]).exists()
    assert Path(summary["tables"]["correlations"]).exists()
    assert Path(summary["tables"]["family"]).exists()


def test_colab_frontend_strips_only_kernel_arguments():
    frontend = _load(
        FRONTEND, "synthetic_scoring_refinement_frontend_test"
    )
    values = frontend._strip_colab_kernel_args(
        [
            "--phase",
            "figures",
            "-f",
            "/root/.local/share/jupyter/runtime/kernel-abc.json",
            "--seeds",
            "0",
            "1",
        ]
    )
    assert values == ["--phase", "figures", "--seeds", "0", "1"]
