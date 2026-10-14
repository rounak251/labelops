"""Behavioral checks for data isolation, calibration math and reproducibility."""

import json

import numpy as np
import pytest

from labelops import benchmark
from labelops.data import load_dataset
from labelops.metrics import classification_metrics
from labelops.model import fit_model, rank_uncertainty


def test_split_partitions_and_seed_labels_are_training_only():
    data = load_dataset()
    train, validation, test = map(set, (data.train_ids, data.validation_ids, data.test_ids))
    assert not train & validation
    assert not train & test
    assert not validation & test
    assert train | validation | test == set(range(len(data.y)))
    assert len(data.seed_ids) == 100
    assert set(data.seed_ids).isdisjoint(data.pool_ids)
    assert set(data.seed_ids) | set(data.pool_ids) == train
    assert set(data.y[data.seed_ids]) == set(range(10))
    repeated = load_dataset()
    assert data.fingerprint == repeated.fingerprint
    assert data.fingerprint != load_dataset(21).fingerprint


def test_scaler_fits_exactly_the_supplied_labeled_rows():
    data = load_dataset()
    rows = data.X[data.seed_ids]
    fitted = fit_model(rows, data.y[data.seed_ids])
    np.testing.assert_allclose(fitted.named_steps["scaler"].mean_, rows.mean(axis=0))
    assert not np.allclose(fitted.named_steps["scaler"].mean_, data.X.mean(axis=0))
    assert fitted.named_steps["scaler"].n_samples_seen_ == 100
    probabilities = fitted.predict_proba(data.X[data.validation_ids])
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    assert fitted.classes_.tolist() == list(range(10))


def test_entropy_orders_uncertainty_and_breaks_ties_by_id():
    probabilities = np.array([[0.5, 0.5], [1.0, 0.0], [0.5, 0.5], [0.8, 0.2]])
    assert rank_uncertainty(probabilities, np.array([9, 2, 3, 5])).tolist() == [3, 9, 5, 2]
    with pytest.raises(ValueError, match="sum to one"):
        rank_uncertainty(np.array([[0.1, 0.1]]), np.array([1]))


def test_known_calibration_and_multiclass_brier_values():
    labels = np.arange(10)
    perfect = classification_metrics(labels, np.eye(10))
    assert perfect["accuracy"] == perfect["macro_f1"] == 1
    assert perfect["ece"] == perfect["brier"] == 0
    assert perfect["confusion_matrix"] == np.eye(10, dtype=int).tolist()
    uniform = classification_metrics(labels, np.full((10, 10), 0.1))
    assert uniform["accuracy"] == pytest.approx(0.1)
    assert uniform["log_loss"] == pytest.approx(np.log(10))
    assert uniform["brier"] == pytest.approx(0.9)
    assert uniform["ece"] == pytest.approx(0)
    assert sum(item["count"] for item in uniform["calibration_bins"]) == 10
    json.dumps(uniform, allow_nan=False)


def test_benchmark_equal_budgets_train_isolation_and_reproducibility(tmp_path, monkeypatch):
    observed_fits = []
    observed_evaluations = []
    original_fit = benchmark.fit_model
    original_metrics = benchmark.classification_metrics

    def capture_fit(X, y, seed=42):
        observed_fits.append((X.copy(), y.copy(), seed))
        return original_fit(X, y, seed)

    def capture_metrics(y, probabilities):
        observed_evaluations.append(y.copy())
        return original_metrics(y, probabilities)

    monkeypatch.setattr(benchmark, "fit_model", capture_fit)
    monkeypatch.setattr(benchmark, "classification_metrics", capture_metrics)
    first = benchmark.run_benchmark(tmp_path / "first", seeds=(7,), rounds=1, batch_size=10)
    dataset = load_dataset(7)
    assert len(observed_fits) == 4
    assert len(observed_evaluations) == 6
    for strategy_index in range(2):
        evaluation_labels = observed_evaluations[3 * strategy_index : 3 * strategy_index + 3]
        np.testing.assert_array_equal(evaluation_labels[0], dataset.y[dataset.validation_ids])
        np.testing.assert_array_equal(evaluation_labels[1], dataset.y[dataset.validation_ids])
        np.testing.assert_array_equal(evaluation_labels[2], dataset.y[dataset.test_ids])
    for strategy_index, strategy in enumerate(benchmark.STRATEGIES):
        run = first["runs"][0]["strategies"][strategy]
        assert [point["labeled_count"] for point in run["learning_curve"]] == [100, 110]
        final_ids = run["final_train_ids"]
        assert set(final_ids).issubset(dataset.train_ids)
        assert not set(final_ids) & set(dataset.validation_ids)
        assert not set(final_ids) & set(dataset.test_ids)
        for round_index, expected_ids in enumerate((dataset.seed_ids, final_ids)):
            observed_X, observed_y, fit_seed = observed_fits[2 * strategy_index + round_index]
            np.testing.assert_array_equal(observed_X, dataset.X[expected_ids])
            np.testing.assert_array_equal(observed_y, dataset.y[expected_ids])
            assert fit_seed == 7
    random_start = first["runs"][0]["strategies"]["random"]["learning_curve"][0]
    entropy_start = first["runs"][0]["strategies"]["entropy"]["learning_curve"][0]
    assert random_start == entropy_start
    second = benchmark.run_benchmark(tmp_path / "second", seeds=(7,), rounds=1, batch_size=10)
    assert first == second
    for filename in (
        "report.json",
        "learning_curve.csv",
        "report.md",
        "learning_curve.svg",
        "confusion_matrix.svg",
    ):
        assert (tmp_path / "first" / filename).read_bytes() == (
            tmp_path / "second" / filename
        ).read_bytes()


def test_benchmark_rejects_unavailable_training_label_budget(tmp_path):
    with pytest.raises(ValueError, match="exceeds the training pool"):
        benchmark.run_benchmark(tmp_path, seeds=(42,), rounds=30, batch_size=40)
