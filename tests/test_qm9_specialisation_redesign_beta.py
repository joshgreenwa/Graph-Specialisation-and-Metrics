from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from graph_specialisation_metrics.carriage.tasks import get_task


ROOT = Path(__file__).resolve().parents[1]
ENTRY = (
    ROOT
    / "experiments/qm9/analysis/qm9_specialisation_redesign_beta_colab.py"
)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ENTRYPOINT = _load(ENTRY, "_qm9_redesign_entrypoint_test")
SHARED = ENTRYPOINT.load_shared_analysis(ROOT)


def test_qm9_entrypoint_reuses_complete_analysis_with_isolated_profile():
    ENTRYPOINT.configure_qm9_profile(SHARED, onehop_vnode=True)
    args = SHARED.parse_args(["--fast-dev-run"])
    cfg = SHARED.make_config(args)

    assert SHARED.ARCHITECTURES == (
        "qm9_gap_dense",
        "qm9_gap_1hop_vnode",
    )
    assert SHARED.DATASET_LABEL == "QM9 HOMO-LUMO gap"
    assert SHARED.FIGURE_PREFIX == "qm9_gap_headline"
    assert str(SHARED.DEFAULT_OUTPUT).endswith(
        "graph_specialisation_metrics/qm9_gap_redesign_headline_v2"
    )
    assert cfg.integrated_atol == pytest.approx(1.0e-4)
    assert cfg.integrated_unconverged_error_cap == pytest.approx(1.0e-3)
    assert cfg.integrated_max_unconverged_fraction == pytest.approx(1.0e-2)
    assert cfg.fingerprint != "zinc-specialisation-headline-v2"
    assert SHARED.create_outputs.__module__ == "_qm9_shared_specialisation_redesign"


def test_qm9_entrypoint_can_select_non_vnode_onehop_checkpoint():
    cleaned, vnode = ENTRYPOINT._strip_qm9_profile_args(
        ["--phase", "scores", "--onehop-no-vnode"]
    )
    assert cleaned == ["--phase", "scores", "--onehop-no-vnode"]
    assert vnode is False
    ENTRYPOINT.configure_qm9_profile(SHARED, onehop_vnode=False)
    assert SHARED.ARCHITECTURES[-1] == "qm9_gap_1hop"
    assert SHARED.DISPLAY["qm9_gap_1hop"] == "1-hop GRIT"


@pytest.mark.parametrize(
    "raw",
    [
        ["-f", "/root/.local/share/jupyter/runtime/kernel-deadbeef.json"],
        ["-f=/root/.local/share/jupyter/runtime/kernel-deadbeef.json"],
    ],
)
def test_qm9_entrypoint_strips_only_injected_colab_kernel_args(raw):
    supplied = [
        "--phase", "scores",
        *raw,
        "--onehop-checkpoint", "/content/model.ckpt",
    ]
    assert ENTRYPOINT._strip_colab_kernel_args(supplied) == [
        "--phase", "scores",
        "--onehop-checkpoint", "/content/model.ckpt",
    ]
    assert ENTRYPOINT._strip_colab_kernel_args(["-f", "ordinary.txt"]) == [
        "-f", "ordinary.txt",
    ]


def test_qm9_vnode_task_replays_attached_training_contract(monkeypatch, tmp_path):
    import GRIT_QM9_gap

    observed = {}

    def fake_apply(repo_dir, note_dir, args):
        observed.update(
            repo_dir=Path(repo_dir),
            note_dir=Path(note_dir),
            attention=args.attention,
            hops=args.hops,
            global_vnode=args.global_vnode,
            expected=GRIT_QM9_gap.expected_param_count(args),
        )

    monkeypatch.setattr(GRIT_QM9_gap, "apply_qm9_patch", fake_apply)
    monkeypatch.setattr(GRIT_QM9_gap, "verify_qm9_model_patch", lambda _repo: None)

    task = get_task("qm9_gap_1hop_vnode")
    task.env_hooks[0](tmp_path)

    assert task.expected_params == 472_833
    assert task.drive_dir == "/content/drive/MyDrive/grit_qm9_gap_1hop_vnode"
    assert task.dataset_dir == "/content/drive/MyDrive/grit_qm9_gap_data"
    assert observed == {
        "repo_dir": tmp_path,
        "note_dir": tmp_path,
        "attention": "khop",
        "hops": 1,
        "global_vnode": True,
        "expected": 472_833,
    }


def test_vnode_patch_maps_graph_major_cache_back_to_model_row_order():
    torch = pytest.importorskip("torch")

    class BatchLike:
        batch = torch.tensor([0, 0, 1, 1, 0, 1])

    class Attention(torch.nn.Module):
        def forward(self, batch):
            h = torch.zeros(6, 1, 1)
            e = torch.zeros(1)
            return h, e

    class Model:
        attn_layers = [Attention()]

    # Cached source is graph-major:
    # graph0(real0, real1, hub), graph1(real0, real1, hub).
    source = torch.arange(6, dtype=torch.float32).reshape(6, 1, 1)
    with SHARED.patched_head(Model(), 0, 0, source):
        patched, _ = Model.attn_layers[0](BatchLike())

    # Model order is all real rows followed by all VNode rows.
    assert patched[:, 0, 0].tolist() == [0, 1, 3, 4, 2, 5]


def test_cross_graph_mismatch_alignment_preserves_vnode_carrier():
    torch = pytest.importorskip("torch")
    events = [
        {
            "corrupt_wv": [np.zeros((4, 1, 1), dtype=np.float32)],
            "mismatch_event": 1,
            "mismatch_alignment": np.asarray([2, 0, 1]),
        },
        {
            "clean_wv": [
                np.asarray([10.0, 20.0, 30.0, 99.0], dtype=np.float32).reshape(4, 1, 1)
            ],
        },
    ]
    value, valid = SHARED.concatenate_mismatch_transport(
        events,
        [0],
        layer=0,
        device=torch.device("cpu"),
    )
    assert valid.tolist() == [True]
    assert value[:, 0, 0].tolist() == [30.0, 10.0, 20.0, 99.0]
