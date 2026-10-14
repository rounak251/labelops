"""Verify label isolation, consensus-only fitting, and cache/audit persistence."""

import json

import joblib
import numpy as np
import pytest

import labelops.service as service_module
from labelops.service import Workbench
from labelops.store import AnnotationStore


@pytest.fixture
def workbench(tmp_path):
    return Workbench(tmp_path / "workspace")


def test_queue_contains_only_training_pool_without_oracle_labels(workbench):
    queue = workbench.queue(limit=len(workbench.dataset.pool_ids))
    assert {row["id"] for row in queue} == set(workbench.dataset.pool_ids)
    expected_keys = {
        "id",
        "image",
        "prediction",
        "confidence",
        "entropy",
        "probabilities",
        "review_count",
        "disputed",
    }
    assert all(set(row) == expected_keys for row in queue)
    assert all(np.asarray(row["image"]).shape == (8, 8) for row in queue)
    assert all(len(row["probabilities"]) == 10 for row in queue)
    assert all(np.isclose(sum(row["probabilities"]), 1) for row in queue)
    entropies = [row["entropy"] for row in queue]
    assert entropies == sorted(entropies, reverse=True)
    json.dumps(queue, allow_nan=False)


@pytest.mark.parametrize("partition", ["seed_ids", "validation_ids", "test_ids"])
def test_annotating_non_pool_ids_is_rejected(workbench, partition):
    sample_id = int(getattr(workbench.dataset, partition)[0])
    with pytest.raises(ValueError, match="training annotation pool"):
        workbench.annotate(sample_id, "Ada", 3)
    assert workbench.store.events() == []


def test_retrain_fits_seed_plus_consensus_labels_and_no_pending_vote(workbench, monkeypatch):
    first_id, pending_id = map(int, workbench.dataset.pool_ids[:2])
    human_label = (int(workbench.dataset.y[first_id]) + 1) % 10
    workbench.annotate(first_id, "Ada", human_label)
    workbench.annotate(first_id, "Grace", human_label)
    workbench.annotate(pending_id, "Ada", 9)
    assert workbench.summary()["pending_consensus_count"] == 1
    calls = []
    actual_fit = service_module.fit_model

    def capture_fit(X, y, seed):
        calls.append((X.copy(), y.copy()))
        return actual_fit(X, y, seed)

    monkeypatch.setattr(service_module, "fit_model", capture_fit)
    summary = workbench.retrain()
    provenance = summary["model_provenance"]
    expected_ids = sorted(list(map(int, workbench.dataset.seed_ids)) + [first_id])
    assert provenance["training_ids"] == expected_ids
    assert provenance["consensus_labels"] == {str(first_id): human_label}
    assert provenance["training_count"] == 101
    assert provenance["seed_count"] == 100
    assert provenance["consensus_count"] == 1
    assert summary["pending_consensus_count"] == 0
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][0], workbench.dataset.X[expected_ids])
    assert calls[0][1][expected_ids.index(first_id)] == human_label
    assert first_id not in {row["id"] for row in workbench.queue(limit=10000)}
    assert pending_id not in provenance["training_ids"]


def test_revised_and_removed_consensus_are_pending_until_refit(workbench):
    sample_id = int(workbench.dataset.pool_ids[0])
    workbench.annotate(sample_id, "Ada", 1)
    workbench.annotate(sample_id, "Grace", 1)
    workbench.retrain()
    workbench.annotate(sample_id, "Ada", 2)
    assert workbench.summary()["pending_consensus_count"] == 1
    queue = workbench.queue(limit=10000)
    row = next(row for row in queue if row["id"] == sample_id)
    assert row["disputed"]
    assert row["review_count"] == 2
    workbench.retrain()
    assert sample_id not in workbench.summary()["model_provenance"]["training_ids"]
    assert workbench.summary()["pending_consensus_count"] == 0
    workbench.annotate(sample_id, "Grace", 2)
    assert workbench.summary()["pending_consensus_count"] == 1


def test_restart_preserves_model_and_votes_without_refitting(workbench, monkeypatch):
    sample_id = int(workbench.dataset.pool_ids[0])
    workbench.annotate(sample_id, "Ada", 7)
    before = workbench.summary()
    probabilities = workbench.model.predict_proba(workbench.dataset.X[:5])

    def forbidden_fit(*args, **kwargs):
        raise AssertionError("a valid saved model should load without fitting")

    monkeypatch.setattr(service_module, "fit_model", forbidden_fit)
    reopened = Workbench(workbench.workspace)
    assert reopened.summary() == before
    assert reopened.store.annotations() == workbench.store.annotations()
    np.testing.assert_array_equal(
        reopened.model.predict_proba(reopened.dataset.X[:5]), probabilities
    )
    assert not list(workbench.workspace.glob(".model-*.joblib"))


def test_evaluation_never_refits_or_changes_model_provenance(workbench, monkeypatch):
    before = workbench.summary()

    def forbidden_fit(*args, **kwargs):
        raise AssertionError("sealed test evaluation must not fit")

    monkeypatch.setattr(service_module, "fit_model", forbidden_fit)
    result = workbench.evaluation()
    assert 0 <= result["accuracy"] <= 1
    assert workbench.summary() == before
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("limit", [True, 0, -1, 1.5, "12"])
def test_queue_rejects_invalid_limits(workbench, limit):
    with pytest.raises(ValueError):
        workbench.queue(limit)


@pytest.mark.parametrize("partition", ["seed_ids", "validation_ids", "test_ids"])
def test_startup_rejects_prepopulated_non_pool_annotations(tmp_path, monkeypatch, partition):
    dataset = service_module.load_dataset()
    workspace = tmp_path / "prepopulated"
    store = AnnotationStore(workspace / "annotations.sqlite3")
    sample_id = int(getattr(dataset, partition)[0])
    store.record(sample_id, "Ada", 7)
    store.record(sample_id, "Grace", 7)

    def forbidden_fit(*args, **kwargs):
        raise AssertionError("Invalid persisted IDs must be rejected before fitting")

    monkeypatch.setattr(service_module, "fit_model", forbidden_fit)
    with pytest.raises(ValueError, match="Stored annotation sample IDs"):
        Workbench(workspace)
    assert not (workspace / "model.joblib").exists()
    assert len(store.events()) == 2


@pytest.mark.parametrize("partition", ["seed_ids", "validation_ids", "test_ids"])
def test_retrain_rejects_direct_store_bypass_before_model_changes(
    workbench, monkeypatch, partition
):
    sample_id = int(getattr(workbench.dataset, partition)[0])
    before_provenance = workbench.summary()["model_provenance"]
    before_model = workbench.model
    before_cache = (workbench.workspace / "model.joblib").read_bytes()
    # Simulate someone writing directly to the audit rather than using annotate().
    workbench.store.record(sample_id, "Ada", 6)
    workbench.store.record(sample_id, "Grace", 6)

    def forbidden_fit(*args, **kwargs):
        raise AssertionError("Invalid persisted IDs must be rejected before fitting")

    monkeypatch.setattr(service_module, "fit_model", forbidden_fit)
    with pytest.raises(ValueError, match="Stored annotation sample IDs"):
        workbench.retrain()
    assert workbench.model is before_model
    assert workbench._provenance == before_provenance
    assert (workbench.workspace / "model.joblib").read_bytes() == before_cache


def test_model_configuration_change_invalidates_saved_model(workbench, monkeypatch):
    calls = []
    actual_fit = service_module.fit_model

    def capture_fit(*args, **kwargs):
        calls.append(True)
        return actual_fit(*args, **kwargs)

    monkeypatch.setattr(service_module, "fit_model", capture_fit)
    monkeypatch.setitem(service_module.MODEL_CONFIG, "C", 0.5)
    reopened = Workbench(workbench.workspace)
    assert calls == [True]
    assert reopened.model.get_params()["classifier__C"] == 0.5
    assert reopened.summary()["model_provenance"]["model_config"]["C"] == 0.5


def test_dependency_version_change_invalidates_saved_model(workbench, monkeypatch):
    calls = []
    actual_fit = service_module.fit_model
    versions = service_module._runtime_versions()
    versions["numpy"] = "different-version"

    def capture_fit(*args, **kwargs):
        calls.append(True)
        return actual_fit(*args, **kwargs)

    monkeypatch.setattr(service_module, "fit_model", capture_fit)
    monkeypatch.setattr(service_module, "_runtime_versions", lambda: versions.copy())
    reopened = Workbench(workbench.workspace)
    assert calls == [True]
    assert reopened.summary()["model_provenance"]["package_versions"] == versions


def test_actual_pipeline_parameter_mismatch_invalidates_saved_model(workbench, monkeypatch):
    cache = workbench.workspace / "model.joblib"
    bundle = joblib.load(cache)
    bundle["model"].set_params(classifier__C=0.25)
    joblib.dump(bundle, cache)
    calls = []
    actual_fit = service_module.fit_model

    def capture_fit(*args, **kwargs):
        calls.append(True)
        return actual_fit(*args, **kwargs)

    monkeypatch.setattr(service_module, "fit_model", capture_fit)
    reopened = Workbench(workbench.workspace)
    assert calls == [True]
    assert reopened.model.get_params()["classifier__C"] == service_module.MODEL_CONFIG["C"]
