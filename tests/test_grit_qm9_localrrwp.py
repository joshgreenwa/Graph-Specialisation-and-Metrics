from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

RUNNER_PATH = (
    Path(__file__).parents[1]
    / "src/graph_specialisation_metrics/grit_patches/qm9_gap.py"
)
RUNNER_SPEC = importlib.util.spec_from_file_location("grit_qm9_gap", RUNNER_PATH)
assert RUNNER_SPEC is not None and RUNNER_SPEC.loader is not None
qm9_gap = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(qm9_gap)


def test_qm9_rrwp_horizon_defaults_to_global() -> None:
    args = qm9_gap.parse_args([])

    assert args.rrwp_horizon == -1
    assert args.name_tag == "QM9Gap.dense.GRITwRRWP"


def test_qm9_local_rrwp_name_and_command_are_isolated(tmp_path: Path) -> None:
    args = qm9_gap.parse_args(
        [
            "--attention",
            "khop",
            "--hops",
            "3",
            "--rrwp-horizon",
            "1",
            "--global-vnode",
            "--drive-dir",
            str(tmp_path),
        ]
    )
    command = qm9_gap.build_training_command(args, tmp_path)

    assert args.name_tag == "QM9Gap.3hop.GRITwRRWP.VNode.LocalRRWP.h1"
    assert command[command.index("posenc_RRWP.local_horizon") + 1] == "1"


@pytest.mark.parametrize("horizon", [-2, 21])
def test_qm9_invalid_rrwp_horizon_is_rejected(horizon: int) -> None:
    with pytest.raises(SystemExit):
        qm9_gap.parse_args(["--rrwp-horizon", str(horizon)])


def test_old_khop_encoder_is_upgraded_to_frozen_support(tmp_path: Path) -> None:
    encoder = tmp_path / "rrwp_encoder.py"
    encoder.write_text(
        """\
        else:
            needed_channels = self.max_hops + 1  # identity + walks of length 1..k
            if self.max_hops < 1 or needed_channels > raw_rrwp_val.size(1):
                raise ValueError(
                    f"max_hops={self.max_hops} requires {needed_channels} RRWP channels; "
                    f"found {raw_rrwp_val.size(1)}."
                )
            reachable = raw_rrwp_val[:, :needed_channels].abs().sum(dim=-1) > 0
            mask_index = rrwp_idx[:, reachable]
""",
        encoding="utf-8",
    )

    assert qm9_gap._upgrade_khop_mask_to_frozen_support(encoder)
    upgraded = encoder.read_text(encoding="utf-8")
    assert 'batch.get("rrwp_attention_edge_index", None)' in upgraded
    assert "Backward-compatible fallback" in upgraded
