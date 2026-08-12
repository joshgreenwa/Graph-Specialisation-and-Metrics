from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from graph_specialisation_metrics import cli


def _write_config(path: Path, experiment: str = "example") -> Path:
    path.write_text(yaml.safe_dump({"experiment": experiment}), encoding="utf-8")
    return path


def test_console_entry_point_returns_success_and_prints_result(monkeypatch, tmp_path, capsys):
    result = tmp_path / "scores.npz"
    experiment = SimpleNamespace(
        score=lambda _config, **_kwargs: result,
    )
    monkeypatch.setattr(cli, "EXPERIMENTS", {"example": experiment})

    returned = cli.main(["score", "--config", str(_write_config(tmp_path / "config.yaml"))])

    assert returned is None
    assert capsys.readouterr().out.strip() == str(result)


def test_run_rejects_multi_job_training_before_work_starts(monkeypatch, tmp_path):
    def unexpected(*_args, **_kwargs):  # pragma: no cover - assertion helper
        raise AssertionError("training must not start")

    experiment = SimpleNamespace(
        jobs=lambda _config, **_kwargs: [{"job": 0}, {"job": 1}],
        train=unexpected,
        score=unexpected,
    )
    monkeypatch.setattr(cli, "EXPERIMENTS", {"example": experiment})

    with pytest.raises(SystemExit, match="requires --job-index"):
        cli.main(["run", "--config", str(_write_config(tmp_path / "config.yaml"))])


def test_command_specific_options_are_not_silently_ignored():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["train", "--config", "config.yaml", "--checkpoint", "model.pt"])
    with pytest.raises(SystemExit):
        parser.parse_args(["score", "--config", "config.yaml", "--job-index", "0"])


def test_run_rejects_checkpoint_with_ignored_job_index(monkeypatch, tmp_path):
    experiment = SimpleNamespace(
        jobs=lambda _config, **_kwargs: [{"job": 0}],
        train=lambda _config, **_kwargs: {},
        score=lambda _config, **_kwargs: tmp_path / "scores.npz",
    )
    monkeypatch.setattr(cli, "EXPERIMENTS", {"example": experiment})

    with pytest.raises(SystemExit, match="omit it when using --checkpoint"):
        cli.main(
            [
                "run",
                "--config",
                str(_write_config(tmp_path / "config.yaml")),
                "--checkpoint",
                str(tmp_path / "model.pt"),
                "--job-index",
                "0",
            ]
        )


def test_job_index_is_rejected_for_single_run_experiment(monkeypatch, tmp_path):
    experiment = SimpleNamespace(score=lambda _config, **_kwargs: tmp_path / "scores.npz")
    monkeypatch.setattr(cli, "EXPERIMENTS", {"example": experiment})

    with pytest.raises(SystemExit, match="does not use job indices"):
        cli.main(
            [
                "run",
                "--config",
                str(_write_config(tmp_path / "config.yaml")),
                "--job-index",
                "0",
            ]
        )
