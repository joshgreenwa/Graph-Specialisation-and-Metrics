"""Model-free postflight tests for isolated score/carriage workers."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from graph_specialisation_metrics.methodology import finalize_measurement_run
from graph_specialisation_metrics.methodology.cache import (
    CacheContract,
    CanonicalCache,
    atomic_json,
    checkpoint_sha256,
)
from graph_specialisation_metrics.methodology.protocol import (
    PROTOCOL_VERSION,
    MethodologyConfig,
    RunSizes,
    SplitManifest,
)
from graph_specialisation_metrics.methodology.runner import run_prepared
from graph_specialisation_metrics.methodology.scores import head_coordinates


def _measurement_config(
    root: Path,
    *,
    task_count: int = 1,
    seeds: tuple[int, ...] = (0, 1, 2),
) -> MethodologyConfig:
    tasks = tuple(f"measurement_task_{position}" for position in range(task_count))
    checkpoint_dir = root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = {}
    for task in tasks:
        for seed in seeds:
            path = checkpoint_dir / f"{task}_seed{seed}.ckpt"
            path.write_bytes(f"checkpoint:{task}:{seed}".encode())
            checkpoints[f"{task}:{seed}"] = str(path)
    return MethodologyConfig(
        output_dir=str(root),
        tasks=tasks,
        train_seeds=seeds,
        phases=("scores", "carriage"),
        sizes=RunSizes(
            discovery_graphs=2,
            causal_graphs=1,
            clean_ablation_graphs=1,
            semantic_donor_graphs=2,
            sources_per_graph=1,
            donors_per_source=1,
        ),
        checkpoints=checkpoints,
        accelerator="cpu",
    )


def _write_worker(
    config: MethodologyConfig,
    task: str,
    seed: int,
    *,
    findings: list[dict[str, object]] | None = None,
) -> None:
    output_dir = config.root / task / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(config.checkpoints[f"{task}:{seed}"])
    digest = checkpoint_sha256(checkpoint)
    geometry = {
        "layers": 1,
        "heads": 2,
        "head_width": 2,
        "hidden_width": 4,
        "outputs": 1,
    }
    splits = SplitManifest(
        discovery=(10, 20),
        causal=(30,),
        clean_ablation=(40,),
        semantic_donor_pool=(50, 60),
        same_index_space=False,
        seed=config.analysis_seed,
    )

    worker_protocol = config.record()
    worker_protocol.update(
        {
            "repository_commit": "worker-commit",
            "execution_mode": "isolated-seed-worker",
            "worker_task": task,
            "worker_seed": seed,
        }
    )
    atomic_json(output_dir / "protocol.json", worker_protocol)
    atomic_json(
        output_dir / "model.json",
        {
            "protocol_version": PROTOCOL_VERSION,
            "repository_commit": "worker-commit",
            "task": task,
            "train_seed": seed,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": digest,
            "task_adapter_version": "test-adapter-v1",
            "output_representation": "scalar",
            "sigma": [1.0],
            "model_geometry": geometry,
            "validation_metric": 0.1,
            "test_metric": 0.2,
            "parameter_count": 123,
            "splits": dataclasses.asdict(splits),
            "canonical_audits": {"failures": []},
        },
    )
    findings = list(findings or [])
    atomic_json(
        output_dir / "audits.json",
        {
            "protocol_version": PROTOCOL_VERSION,
            "task": task,
            "train_seed": seed,
            "strict_audits": config.strict_audits,
            "phases": list(config.phases),
            "findings": findings,
            "headline_eligible": not bool(findings),
        },
    )

    common_contract = {
        "protocol_fingerprint": config.fingerprint,
        "task": task,
        "task_adapter_version": "test-adapter-v1",
        "checkpoint_sha256": digest,
        "train_seed": seed,
        "model_geometry": geometry,
        "output_representation": "scalar",
        "sigma": (1.0,),
        "split_fingerprint": splits.fingerprint,
        "donors_per_source": config.sizes.donors_per_source,
        "source_cap": config.sizes.sources_per_graph,
        "bootstrap_seed": config.bootstrap.rng_seed,
        "bootstrap_replicates": config.bootstrap.replicates,
        "repository_commit": "worker-commit",
    }
    semantic = np.asarray([[1.0 + seed, 2.0 + seed]], dtype=np.float64)
    structural = np.asarray([[2.0 + seed, 1.0 + seed]], dtype=np.float64)
    score_manifest = f"scores:{task}:{seed}"
    score_value = {
        "protocol_version": PROTOCOL_VERSION,
        "manifest_hash": score_manifest,
        "channels": {
            "semantic": {
                "raw": semantic,
                "graph_scores": {graph_id: semantic for graph_id in splits.discovery},
            },
            "structural": {
                "raw": structural,
                "graph_scores": {graph_id: structural for graph_id in splits.discovery},
            },
        },
        "coordinates": head_coordinates(
            semantic,
            structural,
            score_floor=config.numerical.score_floor,
            epsilon=config.numerical.selectivity_epsilon,
            activity_floor=config.families.activity_floor,
        ),
    }
    CanonicalCache(
        config.root,
        CacheContract(event_manifest_hash=score_manifest, **common_contract),
    ).save("scores", "raw", score_value)

    carriage_manifest = f"carriage:{task}:{seed}"
    graph_fields = {
        graph_id: {
            "sources": (0,),
            "donor_counts": (1,),
            "F_sens": np.ones((1, 1), dtype=np.float64),
            "B": np.ones((1, 1), dtype=np.float64),
            "event_B": np.ones((1, 1, 1), dtype=np.float64),
        }
        for graph_id in splits.discovery
    }
    carriage_value = {
        "protocol_version": PROTOCOL_VERSION,
        "manifest_hash": carriage_manifest,
        "channels": {
            channel: {"graph_fields": graph_fields}
            for channel in ("semantic", "structural")
        },
    }
    CanonicalCache(
        config.root,
        CacheContract(event_manifest_hash=carriage_manifest, **common_contract),
    ).save("carriage", "fields", carriage_value)


def _write_complete_run(
    config: MethodologyConfig,
    *,
    findings: dict[tuple[str, int], list[dict[str, object]]] | None = None,
) -> None:
    for task in config.tasks:
        for seed in config.seeds_for(task):
            _write_worker(
                config,
                task,
                seed,
                findings=(findings or {}).get((task, seed)),
            )


def test_measurement_finalizer_validates_ten_tasks_three_seeds_model_free(
    tmp_path,
    monkeypatch,
):
    config = _measurement_config(tmp_path, task_count=10)
    _write_complete_run(config)

    def unexpected_model_load(*_args, **_kwargs):
        raise AssertionError("measurement finalization must not prepare a model or dataset")

    monkeypatch.setattr(
        "graph_specialisation_metrics.methodology.runner.prepare_task",
        unexpected_model_load,
    )
    results = finalize_measurement_run(config)

    assert len(results) == 30
    assert all(result["carriage"] is None for result in results.values())
    assert all(
        set(result["scores"]) <= {
            "channels",
            "coordinates",
            "specialisation_diagnostics",
            "specialist_classification",
        }
        for result in results.values()
    )
    assert all(result["causal"] is None for result in results.values())
    assert all(result["figures"] == {} for result in results.values())
    postflight = json.loads((tmp_path / "measurement_postflight.json").read_text())
    assert postflight["status"] == "complete"
    assert postflight["expected_run_count"] == 30
    assert postflight["validated_run_count"] == 30
    assert len(postflight["tasks"]) == 10
    assert all(
        run["checkpoint_verification"]["status"] == "file_sha256_verified"
        for run in postflight["runs"].values()
    )
    protocol = json.loads((tmp_path / "protocol.json").read_text())
    assert protocol["execution_mode"] == "model-free-measurement-finalizer"
    assert len(protocol["source_cache_contract_fingerprints"]) == 30
    index = json.loads((tmp_path / "index.json").read_text())
    assert index["execution_mode"] == "model-free-measurement-finalizer"
    assert len(index["runs"]) == 30
    assert all(set(run["artifacts"]) == {"scores", "carriage"} for run in index["runs"].values())
    for task in config.tasks:
        population = json.loads((tmp_path / task / "population.json").read_text())
        assert [row["seed"] for row in population["seed_estimates"]] == [0, 1, 2]
        assert population["population_interval"]["level"] == "training seed"


def test_measurement_finalizer_preserves_soft_audit_ineligibility(tmp_path):
    config = _measurement_config(tmp_path, seeds=(0,))
    task = config.tasks[0]
    finding = {"code": "test.soft_finding", "message": "review this run"}
    _write_complete_run(config, findings={(task, 0): [finding]})

    results = finalize_measurement_run(config)

    key = f"{task}:seed0"
    assert results[key]["headline_eligible"] is False
    root_audits = json.loads((tmp_path / "audits.json").read_text())
    assert root_audits["runs"][key] == [finding]
    postflight = json.loads((tmp_path / "measurement_postflight.json").read_text())
    assert postflight["runs"][key]["audit_findings"] == 1
    assert postflight["runs"][key]["headline_eligible"] is False


def test_measurement_finalizer_rejects_incomplete_graph_population(tmp_path):
    config = _measurement_config(tmp_path, seeds=(0,))
    _write_complete_run(config)
    task = config.tasks[0]
    path = tmp_path / task / "seed_0" / "cache" / "scores" / "raw.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["value"]["channels"]["structural"]["graph_scores"].pop(20)
    torch.save(payload, path)

    with pytest.raises(RuntimeError, match="graph population is incomplete"):
        finalize_measurement_run(config)


def test_measurement_finalizer_rejects_partial_files(tmp_path):
    config = _measurement_config(tmp_path, seeds=(0,))
    _write_complete_run(config)
    task = config.tasks[0]
    partial = tmp_path / task / "seed_0" / "cache" / "scores" / "raw.pt.partial"
    partial.write_bytes(b"unfinished")

    with pytest.raises(RuntimeError, match=r"incomplete \.partial files"):
        finalize_measurement_run(config)


def test_measurement_finalizer_rehashes_mounted_checkpoint(tmp_path):
    config = _measurement_config(tmp_path, seeds=(0,))
    _write_complete_run(config)
    checkpoint = Path(config.checkpoints[f"{config.tasks[0]}:0"])
    checkpoint.write_bytes(b"changed after worker completed")

    with pytest.raises(RuntimeError, match="mounted checkpoint.*SHA-256"):
        finalize_measurement_run(config)


def test_measurement_finalizer_requires_worker_audit(tmp_path):
    config = _measurement_config(tmp_path, seeds=(0,))
    _write_complete_run(config)
    task = config.tasks[0]
    (tmp_path / task / "seed_0" / "audits.json").unlink()

    with pytest.raises(FileNotFoundError, match="worker audit"):
        finalize_measurement_run(config)


def test_measurement_finalizer_requires_recomputed_model_metrics(tmp_path):
    config = _measurement_config(tmp_path, seeds=(0,))
    _write_complete_run(config)
    model_path = tmp_path / config.tasks[0] / "seed_0" / "model.json"
    model = json.loads(model_path.read_text(encoding="utf-8"))
    model.pop("validation_metric")
    atomic_json(model_path, model)

    with pytest.raises(RuntimeError, match="parameter count, or metrics"):
        finalize_measurement_run(config)


def test_measurement_finalizer_requires_score_carriage_phase_config(tmp_path):
    config = dataclasses.replace(
        _measurement_config(tmp_path, seeds=(0,)),
        phases=("figures",),
    )

    with pytest.raises(ValueError, match="requires phases"):
        finalize_measurement_run(config)


def test_nonretained_worker_releases_scores_before_carriage(tmp_path, monkeypatch):
    config = _measurement_config(tmp_path, seeds=(0,))
    prepared = SimpleNamespace(
        task=SimpleNamespace(name=config.tasks[0]),
        grit=SimpleNamespace(sc=SimpleNamespace(seed=0)),
        progress=None,
        output_dir=tmp_path / config.tasks[0] / "seed_0",
    )
    order = []
    monkeypatch.setattr(
        "graph_specialisation_metrics.methodology.runner._stage_plan",
        lambda *_args: {0: {}},
    )
    monkeypatch.setattr(
        "graph_specialisation_metrics.methodology.runner.run_scores",
        lambda *_args, **_kwargs: order.append("scores") or {"large": "scores"},
    )
    monkeypatch.setattr(
        "graph_specialisation_metrics.methodology.runner.run_carriage",
        lambda *_args, **_kwargs: order.append("carriage") or {"large": "carriage"},
    )
    monkeypatch.setattr(
        "graph_specialisation_metrics.methodology.runner._release_component_memory",
        lambda component: order.append(f"release:{component}"),
    )

    result = run_prepared(prepared, config, retain_results=False)

    assert result["scores"] is None
    assert result["carriage"] is None
    assert order == [
        "scores",
        "release:scores",
        "carriage",
        "release:carriage",
    ]
