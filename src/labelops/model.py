"""Fixed, train-only preprocessing and logistic classification."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

MODEL_CONFIG = {
    "preprocessor": "StandardScaler",
    "classifier": "LogisticRegression",
    "C": 1.0,
    "solver": "lbfgs",
    "max_iter": 2000,
    "tol": 1e-4,
    "device": "CPU",
    "numerical_threads_during_fit": 1,
}


def fit_model(X: np.ndarray, y: np.ndarray, seed: int = 42) -> Pipeline:
    """Fit only the rows supplied by the caller; no implicit pool access."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y) or len(y) == 0:
        raise ValueError("Training features and labels must be nonempty aligned arrays.")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Training features and labels must be finite.")
    if not np.isin(y, np.arange(10)).all() or len(np.unique(y)) < 2:
        raise ValueError("Training requires at least two classes with labels from 0 to 9.")
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=MODEL_CONFIG["C"],
                    solver=MODEL_CONFIG["solver"],
                    max_iter=MODEL_CONFIG["max_iter"],
                    tol=MODEL_CONFIG["tol"],
                    random_state=int(seed),
                ),
            ),
        ]
    )
    with threadpool_limits(limits=1):
        model.fit(X, y.astype(np.int64))
    return model


def predictive_entropy(probs: np.ndarray) -> np.ndarray:
    """Shannon entropy in nats, with zero probability contributing zero."""
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[1] < 2:
        raise ValueError("Probabilities must be a two-dimensional class matrix.")
    if not np.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any():
        raise ValueError("Probabilities must be finite values between 0 and 1.")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Probability rows must sum to one.")
    return -(probs * np.log(np.maximum(probs, np.finfo(float).tiny))).sum(axis=1)


def rank_uncertainty(probs: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Return IDs by descending predictive entropy, breaking ties by ID."""
    ids = np.asarray(ids, dtype=np.int64)
    entropy = predictive_entropy(probs)
    if ids.ndim != 1 or len(ids) != len(entropy):
        raise ValueError("Sample IDs must align with probability rows.")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Sample IDs must be unique.")
    return ids[np.lexsort((ids, -entropy))]
