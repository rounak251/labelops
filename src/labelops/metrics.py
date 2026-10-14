"""Classification and top-label calibration diagnostics, using classes 0..9."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)

from labelops.model import predictive_entropy


def classification_metrics(y: np.ndarray, probs: np.ndarray) -> dict:
    """Compute JSON-safe metrics; confidence is not promised to be calibrated.

    ECE uses 10 equal-width bins of maximum predicted probability. Multiclass
    Brier is the mean sum of squared error across all 10 class probabilities
    (range 0..2), rather than the mean per class.
    """
    y = np.asarray(y)
    probs = np.asarray(probs, dtype=np.float64)
    if y.ndim != 1 or len(y) == 0 or probs.shape != (len(y), 10):
        raise ValueError(
            "Metrics require nonempty labels and a matching 10-class probability matrix."
        )
    if not np.isin(y, np.arange(10)).all():
        raise ValueError("Evaluation labels must be integers from 0 to 9.")
    predictive_entropy(probs)
    y = y.astype(np.int64)
    prediction = np.argmax(probs, axis=1)
    confidence = probs.max(axis=1)
    correct = prediction == y
    assignments = np.minimum((confidence * 10).astype(int), 9)
    calibration_bins = []
    ece = 0.0
    for index in range(10):
        mask = assignments == index
        count = int(mask.sum())
        bin_confidence = float(confidence[mask].mean()) if count else None
        bin_accuracy = float(correct[mask].mean()) if count else None
        gap = abs(bin_confidence - bin_accuracy) if count else None
        if count:
            ece += count / len(y) * gap
        calibration_bins.append(
            {
                "lower": index / 10,
                "upper": (index + 1) / 10,
                "count": count,
                "mean_confidence": bin_confidence,
                "accuracy": bin_accuracy,
                "absolute_gap": gap,
            }
        )
    precision, recall, f1, support = precision_recall_fscore_support(
        y, prediction, labels=np.arange(10), zero_division=0
    )
    return {
        "sample_count": int(len(y)),
        "accuracy": float(accuracy_score(y, prediction)),
        "macro_f1": float(
            f1_score(y, prediction, labels=np.arange(10), average="macro", zero_division=0)
        ),
        "log_loss": float(log_loss(y, probs, labels=np.arange(10))),
        "ece": float(ece),
        "brier": float(np.square(probs - np.eye(10)[y]).sum(axis=1).mean()),
        "confusion_matrix": confusion_matrix(y, prediction, labels=np.arange(10)).tolist(),
        "calibration_bins": calibration_bins,
        "per_class": [
            {
                "class": int(label),
                "precision": float(precision[label]),
                "recall": float(recall[label]),
                "f1": float(f1[label]),
                "support": int(support[label]),
            }
            for label in range(10)
        ],
    }
