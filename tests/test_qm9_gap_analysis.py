"""QM9 gap registration and Colab-orchestration regression tests (no GRIT run required)."""

from pathlib import Path

import numpy as np
import pytest

from graph_specialisation_metrics.carriage.tasks import get_task, resolve_dataset_dir
from graph_specialisation_metrics.comparison import data as comparison_data
from graph_specialisation_metrics.comparison.plots import _atom_index_label
from graph_specialisation_metrics.comparison.run import DEFAULT_CKPTS
from graph_specialisation_metrics.specialisation.attention_viz import _visual_bond_types
from graph_specialisation_metrics.specialisation.semantic_outlier_ablation import save_attention


QM9_DATA = "/content/drive/MyDrive/grit_qm9_gap_data"


@pytest.mark.parametrize(
    "name,attention,hops,drive,clone",
    [
        (
            "qm9_gap_dense", "dense", 1,
            "/content/drive/MyDrive/grit_qm9_gap_dense",
            "/content/GRIT_qm9_gap_dense",
        ),
        (
            "qm9_gap_1hop", "khop", 1,
            "/content/drive/MyDrive/grit_qm9_gap_1hop_real",
            "/content/GRIT_qm9_gap_1hop",
        ),
    ],
)
def test_qm9_gap_tasks_match_training_contract(name, attention, hops, drive, clone):
    task = get_task(name)
    assert task.config_path == "configs/GRIT/qm9-gap-GRIT-RRWP.yaml"
    assert task.expected_params == 472_769
    assert task.drive_dir == drive
    assert task.dataset_dir == QM9_DATA
    assert task.grit_repo_dir == clone
    assert task.metric_name == "MAE (eV)"
    assert task.metric_higher_better is False
    assert task.metric_abort == 0.5
    assert resolve_dataset_dir(task) == QM9_DATA
    assert len(task.env_hooks) == 1


@pytest.mark.parametrize(
    "name,expected_attention",
    [("qm9_gap_dense", "dense"), ("qm9_gap_1hop", "khop")],
)
def test_qm9_hook_replays_exact_training_patch(name, expected_attention, monkeypatch, tmp_path):
    import GRIT_QM9_gap

    calls = []

    def fake_apply(repo_dir, note_dir, args):
        # Exercise the helper reached by the real patch's provenance writer. This guards the
        # complete Namespace contract, not merely the attributes read directly in its main body.
        expected_count = GRIT_QM9_gap.expected_param_count(args)
        calls.append(
            (
                "apply", Path(repo_dir), Path(note_dir), args.attention, int(args.hops),
                bool(args.global_vnode), int(args.batch_size), int(args.epochs),
                int(args.warmup_epochs), args.expected_params, expected_count,
            )
        )

    monkeypatch.setattr(
        GRIT_QM9_gap,
        "apply_qm9_patch",
        fake_apply,
    )
    monkeypatch.setattr(
        GRIT_QM9_gap,
        "verify_qm9_model_patch",
        lambda repo_dir: calls.append(("verify", Path(repo_dir))),
    )

    get_task(name).env_hooks[0](tmp_path)

    assert calls == [
        (
            "apply", tmp_path, tmp_path, expected_attention, 1, False, 128, 300, 10,
            None, 472_769,
        ),
        ("verify", tmp_path),
    ]


def test_default_qm9_checkpoints_pin_user_identified_best_epochs():
    assert DEFAULT_CKPTS["qm9_gap_dense"].endswith(
        "qm9-gap-GRIT-RRWP-QM9Gap.dense.GRITwRRWP/0/ckpt/294.ckpt")
    assert DEFAULT_CKPTS["qm9_gap_1hop"].endswith(
        "qm9-gap-GRIT-RRWP-QM9Gap.1hop.GRITwRRWP/0/ckpt/295.ckpt")


def test_qm9_comparison_suite_and_dense_role():
    assert comparison_data.QM9_GAP_TASKS == ["qm9_gap_dense", "qm9_gap_1hop"]
    assert comparison_data.is_dense("qm9_gap_dense")
    assert not comparison_data.is_dense("qm9_gap_1hop")
    assert comparison_data.method_meta("qm9_gap_dense")["label"] == "Dense"
    assert comparison_data.method_meta("qm9_gap_1hop")["label"] == "1-hop"


def test_qm9_bond_indices_are_normalised_for_visualisation():
    raw = np.asarray([0, 1, 2, 3])
    assert _visual_bond_types(raw, "PyG-QM9").tolist() == [1, 2, 3, 4]
    assert _visual_bond_types(raw, "PyG-ZINC").tolist() == [0, 1, 2, 3]
    assert _atom_index_label(2, 8, atomic_numbers=True) == "2\nO"
    assert _atom_index_label(2, 8, atomic_numbers=False) == "2"


def test_atomic_number_encoding_survives_attention_cache_roundtrip(tmp_path):
    path = tmp_path / "attention.npz"
    save_attention(
        {
            "heads": [(0, 0)],
            "has_vnode": False,
            "atom_encoding": "atomic_number",
            "molecules": [{
                "graph_id": 7,
                "atom_types": np.asarray([6, 8]),
                "bonds": np.asarray([[0, 1]]),
                "bond_types": np.asarray([2]),
                "pos": np.asarray([[0.0, 0.0], [1.0, 0.0]]),
                "maps": {(0, 0): np.eye(2)},
            }],
        },
        path,
    )
    loaded = comparison_data.load_semantic_outlier_attention(path)
    assert loaded["atom_encoding"] == "atomic_number"
    assert loaded["molecules"][0]["bond_types"].tolist() == [2]


def test_qm9_notebook_cell_is_syntax_valid_and_uses_isolated_cache():
    path = Path(__file__).parents[1] / "experiments/carriage/notebook_qm9_gap_models_cell.py"
    source = path.read_text(encoding="utf-8")
    compile(source, str(path), "exec")
    assert 'QM9_TASKS = ["qm9_gap_dense", "qm9_gap_1hop"]' in source
    assert "graph_specialisation_metrics/qm9_gap" in source
    assert 'dataset_label="QM9 HOMO-LUMO gap"' in source
    assert "ckpt/294.ckpt" in source and "ckpt/295.ckpt" in source
