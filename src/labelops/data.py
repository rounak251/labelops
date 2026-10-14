"""Deterministic dataset splits, retaining original sample IDs."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class Dataset:
    """Arrays are indexed by original digits IDs, never by split row positions."""

    X: np.ndarray
    y: np.ndarray
    train_ids: np.ndarray
    validation_ids: np.ndarray
    test_ids: np.ndarray
    seed_ids: np.ndarray
    pool_ids: np.ndarray
    fingerprint: str
    split_seed: int


def load_dataset(seed: int = 42) -> Dataset:
    """Split digits 60/20/20, then select 100 stratified training seed labels.

    The dataset loader exposes labels for evaluation and the disclosed benchmark
    simulation. The interactive service must never return training-pool labels.
    Pixel scaling is deliberately left to the train-only model pipeline.
    """
    seed = int(seed)
    digits = load_digits()
    X = np.asarray(digits.data, dtype=np.float64)
    y = np.asarray(digits.target, dtype=np.int64)
    ids = np.arange(len(y), dtype=np.int64)
    train_ids, heldout_ids = train_test_split(ids, train_size=0.6, stratify=y, random_state=seed)
    validation_ids, test_ids = train_test_split(
        heldout_ids, test_size=0.5, stratify=y[heldout_ids], random_state=seed
    )
    seed_ids, pool_ids = train_test_split(
        train_ids, train_size=100, stratify=y[train_ids], random_state=seed
    )
    split_arrays = tuple(
        np.sort(part).astype(np.int64)
        for part in (train_ids, validation_ids, test_ids, seed_ids, pool_ids)
    )
    digest = hashlib.sha256()
    for name, array in zip(
        ("X", "y", "train", "validation", "test", "seed", "pool"),
        (X, y, *split_arrays),
        strict=True,
    ):
        digest.update(name.encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(np.ascontiguousarray(array).tobytes())
    return Dataset(X, y, *split_arrays, digest.hexdigest(), seed)


def dataset_provenance(dataset: Dataset) -> dict:
    """Human-readable source and isolation facts for machine-readable reports."""
    return {
        "name": "scikit-learn bundled digits",
        "loader": "sklearn.datasets.load_digits",
        "source_url": "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html",
        "original_source_url": "https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits",
        "samples": int(len(dataset.y)),
        "features": int(dataset.X.shape[1]),
        "image_shape": [8, 8],
        "pixel_range": [0, 16],
        "classes": list(range(10)),
        "split_seed": dataset.split_seed,
        "fingerprint_sha256": dataset.fingerprint,
        "split_counts": {
            "train": int(len(dataset.train_ids)),
            "validation": int(len(dataset.validation_ids)),
            "test": int(len(dataset.test_ids)),
            "initial_labeled": int(len(dataset.seed_ids)),
            "initial_pool": int(len(dataset.pool_ids)),
        },
        "split_method": "stratified 60/20/20 sample split; 100 stratified training seed labels",
        "limitations": [
            "Small educational benchmark, not evidence of production performance.",
            "Bundled samples are from the original UCI test partition, resplit here.",
            "Sample splits are not writer-disjoint; writer identity is unavailable in the loader.",
            "Interactive pool labels are hidden; simulation alone accesses oracle pool labels.",
        ],
    }
