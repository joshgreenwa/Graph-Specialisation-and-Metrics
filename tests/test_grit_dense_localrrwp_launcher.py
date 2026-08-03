from __future__ import annotations

import json
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

RUNNER_PATH = (
    Path(__file__).parents[1] / "experiments/zinc/training/grit_zinc_core.py"
)
RUNNER_SPEC = importlib.util.spec_from_file_location("grit_zinc_core", RUNNER_PATH)
assert RUNNER_SPEC is not None and RUNNER_SPEC.loader is not None
runner = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(runner)


def _write_patch_fixture(repo: Path) -> None:
    files = {
        "grit/config/posenc_config.py": "    cfg.posenc_RRWP.spd = False\n",
        "grit/transform/rrwp.py": (
            "def add_full_rrwp(data,\n"
            "                  spd=False,\n"
            "                  **kwargs):\n"
            "    edge_index, edge_weight = data.edge_index, data.edge_weight\n"
            "    pe = torch.stack(pe_list, dim=-1) # n x n x k\n"
        ),
        "grit/transform/posenc_stats.py": (
            "                            spd=param.spd, # by default False\n"
        ),
    }
    for relative, text in files.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def test_dense_local_rrwp_defaults_are_isolated() -> None:
    args = runner.parse_args(["--rrwp-horizon", "1"])

    assert args.drive_dir == Path("/content/drive/MyDrive/grit_zinc_dense_localrrwp")
    assert args.repo_dir == Path("/content/GRIT_dense_localrrwp")
    assert args.name_tag == "ColabDrive.dense.LocalRRWP.h1.GRITwRRWP"


def test_dense_local_rrwp_command_retains_official_dense_config(tmp_path: Path) -> None:
    args = runner.parse_args(
        [
            "--rrwp-horizon",
            "1",
            "--drive-dir",
            str(tmp_path / "drive"),
            "--repo-dir",
            str(tmp_path / "repo"),
        ]
    )
    command = runner.build_training_command(args, args.drive_dir)

    assert command[command.index("--cfg") + 1] == runner.OFFICIAL_CFG
    assert command[command.index("posenc_RRWP.local_horizon") + 1] == "1"
    assert "gt.attn.full_attn" not in command


@pytest.mark.parametrize("horizon", [-2, 21])
def test_invalid_rrwp_horizon_is_rejected(horizon: int) -> None:
    with pytest.raises(ValueError):
        runner.parse_args(["--rrwp-horizon", str(horizon)])


def test_dense_local_rrwp_patch_is_idempotent(tmp_path: Path) -> None:
    _write_patch_fixture(tmp_path)

    runner.apply_dense_local_rrwp_patch(tmp_path)
    first = {
        path.relative_to(tmp_path): path.read_text(encoding="utf-8")
        for path in tmp_path.rglob("*.py")
    }
    runner.apply_dense_local_rrwp_patch(tmp_path)
    second = {
        path.relative_to(tmp_path): path.read_text(encoding="utf-8")
        for path in tmp_path.rglob("*.py")
    }

    assert second == first
    rrwp = second[Path("grit/transform/rrwp.py")]
    assert "local_horizon=None" in rrwp
    assert "keep_channels = local_horizon + 1 if add_identity else local_horizon" in rrwp
    assert "pe[..., keep_channels:] = 0" in rrwp


def test_missing_spline_conv_is_nonfatal(monkeypatch: pytest.MonkeyPatch) -> None:
    pip_calls: list[list[str]] = []
    command_calls: list[tuple[list[str], dict]] = []
    fake_torch = SimpleNamespace(
        __version__="2.11.0+cu128",
        version=SimpleNamespace(cuda="12.8"),
    )

    monkeypatch.setattr(
        "importlib.import_module",
        lambda name: fake_torch if name == "torch" else None,
    )
    monkeypatch.setattr(runner, "pip_install", lambda args: pip_calls.append(list(args)))

    def fake_run_cmd(command, **kwargs):
        command_calls.append((list(command), kwargs))
        return SimpleNamespace(returncode=1 if "torch-spline-conv" in command else 0)

    monkeypatch.setattr(runner, "run_cmd", fake_run_cmd)
    runner.install_dependencies(
        SimpleNamespace(official_torch112=False, pyg_version="2.2.0")
    )

    core_extensions = next(call for call in pip_calls if "pyg-lib" in call)
    assert "torch-spline-conv" not in core_extensions
    spline_call, spline_kwargs = next(
        (call, kwargs)
        for call, kwargs in command_calls
        if "torch-spline-conv" in call
    )
    assert spline_call
    assert spline_kwargs["check"] is False


def test_dense_localrrwp_notebook_launches_horizon_one() -> None:
    root = Path(__file__).parents[1]
    notebook = root / "experiments/zinc/notebooks/grit_ZINC_dense_localrrwp.ipynb"
    payload = json.loads(notebook.read_text(encoding="utf-8"))
    source = "".join(payload["cells"][0]["source"])
    runner_source = "".join(RUNNER_PATH.read_text(encoding="utf-8").splitlines(keepends=True)[3:])
    expected = runner_source.replace(
        'if __name__ == "__main__":\n    main()\n',
        'if __name__ == "__main__":\n    main(["--rrwp-horizon", "1"])\n',
    )

    assert payload["nbformat"] == 4
    assert len(payload["cells"]) == 1
    assert source == expected
    assert "def apply_dense_local_rrwp_patch" in source


def test_official_notebook_stays_in_sync_with_runner() -> None:
    root = Path(__file__).parents[1]
    notebook = root / "experiments/zinc/notebooks/grit_ZINC_core.ipynb"
    payload = json.loads(notebook.read_text(encoding="utf-8"))
    source = "".join(payload["cells"][0]["source"])
    runner_source = "".join(RUNNER_PATH.read_text(encoding="utf-8").splitlines(keepends=True)[3:])

    assert source == runner_source
    assert source.endswith('if __name__ == "__main__":\n    main()\n')
