from __future__ import annotations

import os
import subprocess
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "graphbench-algoreas-hpc"
    / "bin"
    / "submit_grit_attention_visualisation.sh"
)


def test_attention_submitter_uses_cpu_without_a_gpu_dependency(tmp_path):
    activation = tmp_path / "activate"
    activation.write_text("true\n", encoding="utf-8")
    analysis = tmp_path / "analysis"
    seed_root = analysis / "graphbench_bipartite_matching_hard" / "seed_0"
    score = seed_root / "cache" / "scores" / "raw.pt"
    score.parent.mkdir(parents=True)
    score.write_bytes(b"scores")
    (seed_root / "model.json").write_text("{}\n", encoding="utf-8")
    environment = {
        **os.environ,
        "ENV_ACTIVATE": str(activation),
        "GRAPHBENCH_ANALYSIS_OUTPUT_ROOT": str(analysis),
        "SBATCH_BIN": "/bin/echo",
    }

    result = subprocess.run(
        ["bash", str(SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    assert "--cpus-per-task=2" in result.stdout
    assert "--time=01:00:00" in result.stdout
    assert "--gres" not in result.stdout
    assert "grit_attention_visualisation.sbatch" in result.stdout
    assert "attention_job=" in result.stdout
