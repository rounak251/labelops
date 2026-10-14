"""Workflows for a small, local, human-reviewed active-learning demonstration."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import pickle
import platform
import tempfile
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

from labelops.data import load_dataset
from labelops.metrics import classification_metrics
from labelops.model import MODEL_CONFIG, fit_model, rank_uncertainty
from labelops.store import AnnotationStore, _integer, _votes_to_consensus


def _runtime_versions() -> dict[str, str]:
    """Numerical and serialization dependencies relevant to a cached fit."""
    versions = {
        package: importlib.metadata.version(package)
        for package in ("numpy", "scipy", "scikit-learn", "joblib", "threadpoolctl")
    }
    versions["python"] = platform.python_version()
    return versions


class Workbench:
    """Persist annotations and a fitted model in a trusted local workspace.

    Joblib uses pickle: never point this service at a workspace from an
    untrusted source. Each cache is written to a temporary file, then atomically
    replaced, so a reader sees a complete previous or new model bundle.
    """

    CACHE_SCHEMA = 2

    def __init__(self, workspace: str | Path):
        self.workspace = Path(workspace).expanduser().resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.dataset = load_dataset(seed=42)
        self.store = AnnotationStore(self.workspace / "annotations.sqlite3")
        self._pool_ids = set(map(int, self.dataset.pool_ids))
        self._validated_annotations()
        self._model_config = dict(MODEL_CONFIG)
        self._package_versions = _runtime_versions()
        self._cache_path = self.workspace / "model.joblib"
        self.model = None
        self._provenance: dict = {}
        self._validation_metrics: dict = {}
        if not self._load_model():
            self._fit()
        else:
            self._evaluate_validation()

    def _validated_annotations(self) -> list[dict]:
        annotations = self.store.annotations()
        invalid_ids = sorted({row["sample_id"] for row in annotations} - self._pool_ids)
        if invalid_ids:
            raise ValueError(
                "Stored annotation sample IDs must belong to the training annotation pool; "
                f"invalid IDs: {invalid_ids}"
            )
        return annotations

    def _load_model(self) -> bool:
        if not self._cache_path.exists():
            return False
        try:
            bundle = joblib.load(self._cache_path)
            if (
                bundle["schema_version"] != self.CACHE_SCHEMA
                or bundle["dataset_fingerprint"] != self.dataset.fingerprint
                or bundle["model_config"] != self._model_config
                or bundle["package_versions"] != self._package_versions
            ):
                return False
            provenance = bundle["provenance"]
            seed_ids = sorted(map(int, self.dataset.seed_ids))
            consensus = {int(key): value for key, value in provenance["consensus_labels"].items()}
            if (
                provenance["seed_ids"] != seed_ids
                or not set(consensus).issubset(self._pool_ids)
                or any(
                    isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 9
                    for value in consensus.values()
                )
                or provenance["training_ids"] != sorted(seed_ids + list(consensus))
                or provenance["training_count"] != len(provenance["training_ids"])
                or not isinstance(provenance["model_version"], int)
                or provenance["model_version"] < 1
                or provenance["model_config"] != self._model_config
                or provenance["package_versions"] != self._package_versions
            ):
                return False
            model = bundle["model"]
            if list(map(int, model.classes_)) != list(range(10)):
                return False
            parameters = model.get_params()
            if any(
                parameters[f"classifier__{name}"] != self._model_config[name]
                for name in ("C", "solver", "max_iter", "tol")
            ) or (
                parameters["classifier__random_state"] != 42
                or parameters["scaler__with_mean"] is not True
                or parameters["scaler__with_std"] is not True
                or type(model.named_steps["scaler"]).__name__ != self._model_config["preprocessor"]
                or type(model.named_steps["classifier"]).__name__
                != self._model_config["classifier"]
            ):
                return False
            self.model, self._provenance = model, provenance
            return True
        except (
            OSError,
            EOFError,
            KeyError,
            ValueError,
            TypeError,
            AttributeError,
            ImportError,
            pickle.UnpicklingError,
        ):
            # A missing/incompatible/corrupt local cache is recoverable from the audit.
            return False

    def _persist_model(self) -> None:
        bundle = {
            "schema_version": self.CACHE_SCHEMA,
            "dataset_fingerprint": self.dataset.fingerprint,
            "model_config": self._model_config,
            "package_versions": self._package_versions,
            "model": self.model,
            "provenance": self._provenance,
        }
        handle, temporary = tempfile.mkstemp(prefix=".model-", suffix=".joblib", dir=self.workspace)
        os.close(handle)
        try:
            joblib.dump(bundle, temporary, compress=3)
            with open(temporary, "rb") as fitted_cache:
                os.fsync(fitted_cache.fileno())
            os.replace(temporary, self._cache_path)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def _fit(self) -> None:
        # Validate persisted rows as well as requests: a prepopulated local
        # database must never put validation/test/seed annotations into a fit.
        consensus = _votes_to_consensus(self._validated_annotations())
        model_config = dict(MODEL_CONFIG)
        package_versions = _runtime_versions()
        # Ground-truth labels are accessed only for the fixed initial seed.
        labels = {int(index): int(self.dataset.y[index]) for index in self.dataset.seed_ids}
        labels.update(consensus)
        training_ids = sorted(labels)
        training_labels = np.asarray([labels[index] for index in training_ids], dtype=np.int64)
        self.model = fit_model(self.dataset.X[training_ids], training_labels, seed=42)
        self._model_config = model_config
        self._package_versions = package_versions
        version = self._provenance.get("model_version", 0) + 1
        label_digest = hashlib.sha256(
            json.dumps(
                list(zip(training_ids, map(int, training_labels))), separators=(",", ":")
            ).encode()
        ).hexdigest()
        self._provenance = {
            "model_version": version,
            "dataset_fingerprint": self.dataset.fingerprint,
            "model_config": self._model_config.copy(),
            "package_versions": self._package_versions.copy(),
            "fitted_at": datetime.now(timezone.utc).isoformat(),
            "seed": 42,
            "seed_ids": sorted(map(int, self.dataset.seed_ids)),
            "training_ids": training_ids,
            "consensus_labels": {
                str(index): int(label) for index, label in sorted(consensus.items())
            },
            "training_labels_sha256": label_digest,
            "training_count": len(training_ids),
            "label_count": len(training_ids),
            "seed_count": len(self.dataset.seed_ids),
            "consensus_count": len(consensus),
            "fit_scope": "initial training seed plus current reviewer consensus",
        }
        self._evaluate_validation()
        self._persist_model()

    def _evaluate_validation(self) -> None:
        validation = self.dataset.validation_ids
        self._validation_metrics = classification_metrics(
            self.dataset.y[validation], self.model.predict_proba(self.dataset.X[validation])
        )

    def queue(self, limit: int = 12) -> list[dict]:
        limit = _integer(limit, "limit", 1, 2**31 - 1)
        with self._lock:
            annotations = self.store.annotations()
            consensus = _votes_to_consensus(annotations)
            ids = np.asarray(sorted(self._pool_ids - consensus.keys()), dtype=np.int64)
            if not len(ids):
                return []
            probabilities = self.model.predict_proba(self.dataset.X[ids])
            order = rank_uncertainty(probabilities, ids)[:limit]
            positions = {int(sample_id): index for index, sample_id in enumerate(ids)}
            review_counts = Counter(row["sample_id"] for row in annotations)
            results = []
            for sample_id in order:
                sample_id = int(sample_id)
                probs = probabilities[positions[sample_id]]
                positive = probs[probs > 0]
                results.append(
                    {
                        "id": sample_id,
                        "image": self.dataset.X[sample_id].reshape(8, 8).tolist(),
                        "prediction": int(np.argmax(probs)),
                        "confidence": float(np.max(probs)),
                        "entropy": float(-np.sum(positive * np.log(positive))),
                        "probabilities": probs.tolist(),
                        "review_count": review_counts[sample_id],
                        "disputed": review_counts[sample_id] >= 2 and sample_id not in consensus,
                    }
                )
            return results

    def annotate(self, sample_id: int, reviewer: str, label: int, note: str = "") -> dict:
        sample_id = _integer(sample_id, "sample_id", 0, 2**63 - 1)
        if sample_id not in self._pool_ids:
            raise ValueError("sample_id must belong to the training annotation pool")
        with self._lock:
            return self.store.record(sample_id, reviewer, label, note)

    def retrain(self) -> dict:
        with self._lock:
            self._fit()
            return self.summary()

    def summary(self) -> dict:
        with self._lock:
            current = self.store.consensus()
            fitted = {
                int(index): label for index, label in self._provenance["consensus_labels"].items()
            }
            pending = sum(
                current.get(index) != fitted.get(index) for index in current.keys() | fitted.keys()
            )
            return {
                "dataset_counts": {
                    "total": len(self.dataset.y),
                    "train": len(self.dataset.train_ids),
                    "validation": len(self.dataset.validation_ids),
                    "test": len(self.dataset.test_ids),
                    "seed": len(self.dataset.seed_ids),
                    "pool": len(self.dataset.pool_ids),
                },
                "labeled_count": len(self.dataset.seed_ids) + len(current),
                "fitted_labeled_count": self._provenance["training_count"],
                "pending_consensus_count": pending,
                "agreement": self.store.agreement(),
                "validation_metrics": json.loads(json.dumps(self._validation_metrics)),
                "model_version": self._provenance["model_version"],
                "model_provenance": json.loads(json.dumps(self._provenance)),
            }

    def evaluation(self) -> dict:
        """Explicit sealed-test evaluation; this does not fit or choose samples."""
        with self._lock:
            test = self.dataset.test_ids
            return classification_metrics(
                self.dataset.y[test], self.model.predict_proba(self.dataset.X[test])
            )
