from pathlib import Path

from graph_specialisation_metrics.main_procedure import DEFAULT_CONFIG as MAIN_DEFAULT
from graph_specialisation_metrics.main_procedure import deep_update as main_deep_update
from graph_specialisation_metrics.main_procedure import run_main
from graph_specialisation_metrics.method_validation import DEFAULT_CONFIG as VALIDATION_DEFAULT
from graph_specialisation_metrics.method_validation import deep_update as validation_deep_update
from graph_specialisation_metrics.method_validation import run_all


def test_method_validation_tiny_smoke_writes_four_figures(tmp_path: Path):
    cfg = validation_deep_update(
        VALIDATION_DEFAULT,
        {
            "artifact_root": str(tmp_path / "validation"),
            "device": "cpu",
            "carriage": {
                "num_graphs": 8,
                "num_nodes": 10,
                "analysis_graphs": 2,
                "ig_steps": 3,
                "train_epochs": 2,
                "small_gt": {"layers": 1, "hidden_dim": 16, "heads": 4},
            },
            "patching": {"chain_lengths": [4]},
            "rank": {
                "matrix_size": 12,
                "planted_ranks": [1, 2],
                "noise_levels": [0.05],
                "matrices_per_setting": 2,
                "null_permutations": 2,
            },
            "interaction": {"pairs": 16},
            "figures": {"bootstrap_draws": 20, "dpi": 80},
        },
    )
    root = run_all(cfg, force=True)
    assert (root / "manifest.json").exists()
    assert (root / "figures" / "validation_carriage_check.png").exists()
    assert (root / "figures" / "validation_patching_check.png").exists()
    assert (root / "figures" / "validation_rank_check.png").exists()
    assert (root / "figures" / "validation_interaction_check.png").exists()


def test_main_procedure_dry_run_discovers_and_writes_status(tmp_path: Path):
    model_root = tmp_path / "dense"
    model_root.mkdir()
    (model_root / "stats.json").write_text('{"test_mae": 0.1, "history": [{"epoch": 1, "train_loss": 1.0, "val_loss": 1.2}]}')
    cfg = main_deep_update(
        MAIN_DEFAULT,
        {
            "artifact_root": str(tmp_path / "main"),
            "models": {
                "dense_grit": {"artifact_root": str(model_root)},
                "grit_1hop": {"artifact_root": str(tmp_path / "onehop")},
                "gin": {"artifact_root": str(tmp_path / "gin")},
                "gcn": {"artifact_root": str(tmp_path / "gcn")},
            },
        },
    )
    root = run_main(cfg, steps=["0", "1"], dry_run=True, force=True)
    assert (root / "manifest.json").exists()
    assert (root / "metrics" / "artifact_discovery.json").exists()
    assert (root / "metrics" / "step0_status.json").exists()
    assert (root / "metrics" / "step1_status.json").exists()
