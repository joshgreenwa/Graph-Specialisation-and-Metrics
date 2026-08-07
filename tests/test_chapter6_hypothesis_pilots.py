from __future__ import annotations

import numpy as np
import pytest

from graph_specialisation_metrics.chapter6_hypothesis_pilots import (
    _aligned_fraction,
    _bootstrap_summary,
    run,
)


def test_vnode_mediation_fraction_has_matched_direction():
    clean = np.asarray([0.0, 2.0])
    event = np.asarray([1.0, 0.0])
    restored = np.asarray([0.25, 1.5])
    injected = np.asarray([0.75, 0.5])

    assert _aligned_fraction(clean, event, restored, "restoration") == pytest.approx([0.75, 0.75])
    assert _aligned_fraction(clean, event, injected, "injection") == pytest.approx([0.75, 0.75])


def test_pilot_bootstrap_averages_events_within_graph_first():
    rows = [
        {"task": "zinc", "graph": 0, "value": 1.0},
        {"task": "zinc", "graph": 0, "value": 3.0},
        {"task": "zinc", "graph": 1, "value": 6.0},
    ]
    result = _bootstrap_summary(
        rows,
        groups=("task",),
        metric="value",
        replicates=100,
        seed=0,
    )
    assert result[0]["graphs"] == 2
    assert result[0]["mean"] == pytest.approx(4.0)
    assert result[0]["low"] <= result[0]["mean"] <= result[0]["high"]


def test_missing_pilot_artifacts_skip_without_blocking(tmp_path):
    result = run([tmp_path / "missing"], tmp_path / "output", verbose=False)
    assert not result["figures"]
    assert len(result["warnings"]) == 5
    assert (tmp_path / "output/summary.json").is_file()
