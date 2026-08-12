from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from graph_specialisation_metrics.experiments import ExperimentSetupError, mixed
from graph_specialisation_metrics.experiments._grit import grit_transformer_layer

ROOT = Path(__file__).resolve().parents[1]


def _settings(*, fast: bool = False) -> mixed.Settings:
    config = yaml.safe_load((ROOT / "configs" / "mixed.yaml").read_text(encoding="utf-8"))
    return mixed._settings(config, fast)


def test_mixed_config_matches_dissertation() -> None:
    settings = _settings()

    assert settings.early_stop_checks == 10
    assert settings.early_stop_accuracy == 0.995
    assert settings.acceptance_accuracy == 0.90


def test_structural_source_is_drawn_from_every_other_node(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(fast=True)
    generator = np.random.default_rng(7)
    source_pools: list[np.ndarray] = []

    class RecordingGenerator:
        def __getattr__(self, name: str):
            return getattr(generator, name)

        def choice(self, values, *args, **kwargs):
            source_pools.append(np.asarray(values))
            return generator.choice(values, *args, **kwargs)

    monkeypatch.setattr(mixed.np.random, "default_rng", lambda _seed: RecordingGenerator())
    batch = mixed._make_batch(settings, 8, seed=1, mode=mixed.STRUCTURAL)

    assert len(source_pools) == len(batch)
    for graph, pool in enumerate(source_pools):
        query = int(batch.query[graph])
        assert set(pool.tolist()) == set(range(settings.nodes)) - {query}
        assert int(batch.source[graph]) != query
        assert (
            int(batch.target[graph])
            == mixed._cycle_distance(settings.nodes, query, int(batch.source[graph])) - 1
        )


def test_checkpoint_validation_allows_new_scoring_counts_only() -> None:
    saved = _settings()
    mixed._validate_checkpoint_settings(
        replace(saved, score_graphs=4, donor_swaps_per_source=2, score_seed=99), saved
    )

    with pytest.raises(ExperimentSetupError, match="hidden_dim"):
        mixed._validate_checkpoint_settings(replace(saved, hidden_dim=192), saved)


def test_scoring_uses_identical_donor_swap_streams_across_model_seeds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    torch = pytest.importorskip("torch")
    config = yaml.safe_load((ROOT / "configs" / "mixed.yaml").read_text(encoding="utf-8"))
    settings = mixed._settings(config, fast=True)
    paths = [tmp_path / "seed_0.pt", tmp_path / "seed_1.pt"]
    payloads = {
        path: {
            "settings": settings.__dict__,
            "seed": seed,
            "state_dict": {"seed": seed},
        }
        for seed, path in enumerate(paths)
    }
    monkeypatch.setattr(mixed, "_checkpoint_paths", lambda *_args: paths)
    monkeypatch.setattr(torch, "load", lambda path, **_kwargs: payloads[path])

    class Model:
        def load_state_dict(self, state):
            self.seed = int(state["seed"])

        def to(self, _device):
            return self

        def eval(self):
            return self

    monkeypatch.setattr(mixed, "_model", lambda _settings: Model())
    draws: dict[int, list[int]] = {0: [], 1: []}

    def fake_score(model, used_settings, _clean, _channel, rng, _device):
        draws[model.seed].append(int(rng.integers(0, 2**31)))
        score = np.zeros((used_settings.layers, used_settings.heads))
        distance_contributions = np.zeros((*score.shape, used_settings.nodes))
        return score, distance_contributions

    monkeypatch.setattr(mixed, "_score_graph", fake_score)

    mixed.score(config, checkpoint=tmp_path, output_dir=tmp_path / "results", fast=True)

    assert draws[0] == draws[1]


def test_scoring_rejects_duplicate_checkpoint_seeds_before_model_loading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    torch = pytest.importorskip("torch")
    config = yaml.safe_load((ROOT / "configs" / "mixed.yaml").read_text(encoding="utf-8"))
    settings = mixed._settings(config, fast=True)
    paths = [tmp_path / "first.pt", tmp_path / "second.pt"]
    payload = {"settings": settings.__dict__, "seed": 0, "state_dict": {}}
    monkeypatch.setattr(mixed, "_checkpoint_paths", lambda *_args: paths)
    monkeypatch.setattr(torch, "load", lambda *_args, **_kwargs: payload)
    monkeypatch.setattr(
        mixed,
        "_model",
        lambda _settings: pytest.fail("duplicate seeds must fail before model loading"),
    )

    with pytest.raises(ExperimentSetupError, match="duplicate mixed checkpoint seed 0"):
        mixed.score(config, checkpoint=tmp_path, output_dir=tmp_path / "results", fast=True)


def test_grit_capture_preserves_the_layer_forward() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    try:
        grit_transformer_layer()
    except ExperimentSetupError as exc:
        pytest.skip(str(exc))

    settings = mixed.Settings(
        nodes=4,
        key_vocab=4,
        classes=2,
        rrwp_steps=3,
        layers=1,
        heads=2,
        hidden_dim=8,
        dropout=0.0,
        attention_dropout=0.0,
        seeds=(0,),
        batch_size=2,
        steps=1,
        learning_rate=0.001,
        weight_decay=0.0,
        validation_graphs=2,
        validate_every=1,
        score_graphs=1,
        donor_swaps_per_source=1,
        score_seed=3,
    )
    model = mixed._model(settings).eval()
    batch = mixed._make_batch(settings, 1, seed=4, mode=mixed.SEMANTIC)

    ordinary = model(batch)
    model_output, head_outputs = model(batch, capture=True)

    torch.testing.assert_close(model_output, ordinary)
    assert torch.equal(model._batch(batch).deg, torch.full((settings.nodes,), 2.0))
    gradient = torch.autograd.grad(model_output[0, 0], head_outputs)[0]
    assert torch.isfinite(gradient).all()
    assert float(gradient.abs().sum()) > 0
