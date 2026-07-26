from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from graph_specialisation_metrics.methodology.cache import (
    StaleCacheError,
    load_cache_artifact_file,
)
from graph_specialisation_metrics.methodology.protocol import (
    PROTOCOL_VERSION,
    FamilyPolicy,
    stable_hash,
)
from graph_specialisation_metrics.synthetic.nar_methodology_extension import (
    ExtensionContract,
    ExtensionStore,
    _safe_extension_layout,
    counterfactual_estimand,
    freeze_role_families,
    role_coordinates,
)


def test_role_coordinates_have_interpretable_extremes_and_tv_separation():
    query = np.asarray([[2.0, 0.0], [1.0, 1.0]])
    value = np.asarray([[0.0, 2.0], [1.0, 1.0]])
    result = role_coordinates(
        query,
        value,
        score_floor=1e-12,
        epsilon=1e-12,
        activity_floor=0.2,
    )

    assert result.estimable
    assert result.selectivity[0, 0] == pytest.approx(1.0)
    assert result.selectivity[0, 1] == pytest.approx(-1.0)
    assert result.selectivity[1, 0] == pytest.approx(0.0)
    assert 0.0 <= result.role_separation <= 1.0
    assert result.role_separation == pytest.approx(0.5)


def test_counterfactual_estimand_is_symmetric_and_fail_closed():
    mediation, raw, estimable = counterfactual_estimand(
        clean_margin=[-2.0, -1.0, -1.0],
        counterfactual_margin=[2.0, -2.0, 1.0],
        injected_margin=[0.0, 0.0, 0.0],
        restored_margin=[0.0, 0.0, 0.0],
        label_changed=[True, True, False],
        effect_floor=1e-8,
    )

    assert raw[0] == pytest.approx(2.0)
    assert mediation[0] == pytest.approx(0.5)
    assert estimable.tolist() == [True, False, False]
    assert np.isnan(mediation[1:]).all()


def test_role_families_are_frozen_from_selectivity_with_matched_controls():
    query = np.asarray(
        [
            [5.0, 4.0, 2.0, 1.8, 1.0, 0.8, 0.5, 0.4],
            [4.5, 3.5, 2.2, 1.7, 1.1, 0.7, 0.6, 0.3],
        ]
    )
    value = np.asarray(
        [
            [0.4, 0.5, 1.8, 2.0, 4.0, 5.0, 0.7, 0.6],
            [0.5, 0.6, 1.7, 2.1, 3.5, 4.5, 0.8, 0.7],
        ]
    )
    role = role_coordinates(
        query,
        value,
        score_floor=1e-12,
        epsilon=1e-12,
        activity_floor=0.2,
    )
    families = freeze_role_families(
        role,
        np.ones_like(query),
        policy=FamilyPolicy(
            activity_floor=0.2,
            tail_fraction=0.2,
            central_fraction=0.2,
        ),
        rng_seed=7,
    )

    assert families["address"]
    assert families["content"]
    assert len(families["address_control"]) == len(families["address"])
    assert len(families["content_control"]) == len(families["content"])
    assert set(families["address"]).isdisjoint(families["content"])
    assert all(role.selectivity[head] > 0 for head in families["address"])
    assert all(role.selectivity[head] < 0 for head in families["content"])


def test_extension_store_refuses_an_existing_different_contract(tmp_path):
    store = ExtensionStore(tmp_path / "extension")
    base = ExtensionContract(
        stage="roles",
        task="nar_dense_N4",
        seed=0,
        checkpoint_sha256="checkpoint",
        source_artifact_sha256="source",
        source_contract_fingerprint="contract",
        scientific_parameters={"tail": 0.2},
    )
    store.save(base, "role_scores", {"value": 1})
    changed = dataclasses.replace(
        base, scientific_parameters={"tail": 0.25}
    )

    with pytest.raises(StaleCacheError, match="another scientific contract"):
        store.load(changed, "role_scores")

    assert store.load(base, "role_scores") == {"value": 1}


def test_extension_layout_cannot_overlap_canonical_tree(tmp_path):
    analysis = tmp_path / "analysis"
    canonical = analysis / "canonical"
    canonical.mkdir(parents=True)

    with pytest.raises(ValueError, match="never the canonical cache tree"):
        _safe_extension_layout(analysis, canonical / "extension")

    allowed = analysis / "extensions" / "nar_role_counterfactual_v1"
    _safe_extension_layout(analysis, allowed)


def test_read_only_artifact_loader_accepts_old_repository_provenance_but_checks_itself(
    tmp_path,
):
    torch = pytest.importorskip("torch")
    contract = {
        "task": "nar_dense_N4",
        "train_seed": 0,
        "repository_commit": "an-older-pushed-commit",
    }
    path = tmp_path / "raw.pt"
    torch.save(
        {
            "metadata": {
                "protocol_version": PROTOCOL_VERSION,
                "contract": contract,
                "contract_fingerprint": stable_hash(contract),
            },
            "value": {"score": np.ones((2, 2))},
        },
        path,
    )

    artifact = load_cache_artifact_file(path)
    assert artifact.metadata["contract"]["repository_commit"] == "an-older-pushed-commit"
    assert artifact.value["score"].shape == (2, 2)
    assert len(artifact.file_sha256) == 64

    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["metadata"]["contract_fingerprint"] = "corrupt"
    torch.save(payload, path)
    with pytest.raises(StaleCacheError, match="internally inconsistent"):
        load_cache_artifact_file(path)
