import sys
from pathlib import Path

import numpy as np

from graph_specialisation_metrics import progress
from graph_specialisation_metrics.carriage import env


def test_progress_roundtrip_and_fingerprint_rejection(tmp_path):
    path = tmp_path / "state.npz"
    fingerprint = {"task": "zinc", "graphs": [1, 3], "donors": 4}
    rng = np.random.default_rng(7)
    rng.random(3)
    progress.save_progress(
        path, fingerprint=fingerprint, next_index=1,
        rng_state=rng.bit_generator.state,
        arrays={"x": np.asarray([1.0, 2.0])}, scalars={"count": 3},
    )

    loaded = progress.load_progress(path, fingerprint=fingerprint)
    assert loaded is not None
    assert loaded["next_index"] == 1
    assert loaded["scalars"] == {"count": 3}
    np.testing.assert_array_equal(loaded["arrays"]["x"], [1.0, 2.0])
    restored = np.random.default_rng()
    restored.bit_generator.state = loaded["rng_state"]
    assert restored.random() == rng.random()
    assert progress.load_progress(path, fingerprint={**fingerprint, "donors": 8}) is None


def test_prepare_inprocess_grit_reprioritises_existing_clone(tmp_path, monkeypatch):
    repo = tmp_path / "GRIT_task"
    package = repo / "grit"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("ORIGIN = 'task'\n", encoding="utf-8")

    monkeypatch.setattr(env, "apply_compat_patches", lambda: sys.path.insert(0, "/fake/site"))
    monkeypatch.setattr(env, "run_cmd", lambda *args, **kwargs: None)
    monkeypatch.chdir(tmp_path)
    sys.path.append(str(repo))
    try:
        env.prepare_inprocess_grit(repo)
        assert Path(sys.path[0]).resolve() == repo.resolve()
    finally:
        sys.path[:] = [entry for entry in sys.path if entry not in {str(repo), "/fake/site"}]
        sys.modules.pop("grit", None)
