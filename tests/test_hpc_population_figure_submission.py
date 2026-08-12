from __future__ import annotations

import os
import subprocess
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "graphbench-algoreas-hpc"
    / "bin"
    / "submit_cached_grit_population_figures.sh"
)


def _complete_cache(root: Path) -> None:
    for seed in range(4):
        seed_root = root / "graphbench_bipartite_matching_hard" / f"seed_{seed}"
        for relative in (
            Path("cache/scores/raw.pt"),
            Path("cache/causal/validation.pt"),
            Path("audits.json"),
        ):
            path = seed_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"complete")


def _environment(tmp_path: Path) -> dict[str, str]:
    activation = tmp_path / "activate"
    activation.write_text("true\n", encoding="utf-8")
    return {
        **os.environ,
        "ENV_ACTIVATE": str(activation),
        "GRAPHBENCH_OUTPUT_BASE": str(tmp_path / "outputs"),
        "SBATCH_BIN": "/bin/echo",
    }


def test_cached_figure_submitter_replaces_an_empty_requested_root(tmp_path):
    complete = tmp_path / "outputs" / "actual_analysis"
    _complete_cache(complete)
    requested = tmp_path / "outputs" / "empty_default"
    requested.mkdir(parents=True)
    environment = _environment(tmp_path)
    environment["GRAPHBENCH_ANALYSIS_OUTPUT_ROOT"] = str(requested)

    result = subprocess.run(
        ["bash", str(SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    assert f"requested root is incomplete: {requested}" in result.stdout
    assert f"analysis_root={complete}" in result.stdout
    assert "--cpus-per-task=1" in result.stdout
    assert f"PROJECT_ROOT={SCRIPT.parents[2]}" in result.stdout
    assert "--gres" not in result.stdout


def test_cached_figure_submitter_refuses_ambiguous_complete_roots(tmp_path):
    first = tmp_path / "outputs" / "analysis_a"
    second = tmp_path / "outputs" / "analysis_b"
    _complete_cache(first)
    _complete_cache(second)

    result = subprocess.run(
        ["bash", str(SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        env=_environment(tmp_path),
    )

    assert result.returncode == 2
    assert "Multiple complete matching cache roots were found" in result.stderr
    assert str(first) in result.stderr
    assert str(second) in result.stderr
    assert "cpu_finalizer=" not in result.stdout


def test_cached_figure_submitter_uses_explicit_complete_root_when_several_exist(
    tmp_path,
):
    first = tmp_path / "outputs" / "analysis_a"
    second = tmp_path / "outputs" / "analysis_b"
    _complete_cache(first)
    _complete_cache(second)
    environment = _environment(tmp_path)
    environment["GRAPHBENCH_ANALYSIS_OUTPUT_ROOT"] = str(second)

    result = subprocess.run(
        ["bash", str(SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    assert f"analysis_root={second}" in result.stdout
    assert str(first) not in result.stdout


def test_cached_figure_submitter_prefers_latest_registered_complete_root(tmp_path):
    older = tmp_path / "outputs" / "older_analysis"
    latest = (
        tmp_path
        / "outputs"
        / "grit_specialisation_bipartite_complete_pe_causal_v3"
    )
    _complete_cache(older)
    _complete_cache(latest)

    result = subprocess.run(
        ["bash", str(SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        env=_environment(tmp_path),
    )

    assert result.returncode == 0, result.stderr
    assert f"analysis_root={latest}" in result.stdout
    assert str(older) not in result.stdout
